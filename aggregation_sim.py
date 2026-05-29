import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

# --- SIMÜLASYON PARAMETRELERI ---
GRID_SIZE = 100
NUM_SENSORS = 50
SIMULATION_TIME = 60  # Saniye

# --- ENERJI PARAMETRELERI (milliJoule - mJ) ---
INITIAL_ENERGY = 100.0   # 100 mJ Başlangıç Bataryası
E_SENSE = 0.5            # Ölçüm yapma tüketimi
E_TX = 2.0               # Veri gönderme tüketimi
E_RX = 1.0               # Veri alma tüketimi
E_AGG = 1.5              # CH için veri birleştirme tüketimi

# --- YANGIN VE RÜZGAR PARAMETRELERI ---
IGNITION_POINT = (30, 30)  # Yangının başlama noktası
WIND_VECTOR = (1.5, 0.8)  # Rüzgarın X ve Y yönündeki şiddeti (Kuzeydoğu yönü)
FIRE_SPREAD_RATE = 1.2    # Yangının zamanla büyüme katsayısı
BASE_TEMP = 20.0          # Normal ortam sıcaklığı
FIRE_TEMP_MAX = 800.0     # Merkezdeki maksimum yangın sıcaklığı


class WSN_Simulation:
    def __init__(self, env):
        self.env = env
        self.total_packets_sent_raw = 0
        self.total_packets_sent_agg = 0
        self.total_energy_consumed_raw = 0.0  # Birleştirme olmasaydı ağın harcayacağı enerji
        self.total_energy_consumed_agg = 0.0  # Birleştirme ile ağın gerçekte harcadığı enerji

        self.ch_buffer = []
        self.nodes = {}
        self.history = []
        self.cluster_head_id = "CH"

        # Grid X, Y matrisleri (Isı haritası için)
        x = np.linspace(0, GRID_SIZE, GRID_SIZE)
        y = np.linspace(0, GRID_SIZE, GRID_SIZE)
        self.X, self.Y = np.meshgrid(x, y)

    def init_network(self):
        """Ağı ve sensörleri başlat."""
        # Küme Başı (Merkezde) - Sınırsız veya yüksek enerjili varsayımı
        self.nodes[self.cluster_head_id] = {
            'pos': (GRID_SIZE / 2, GRID_SIZE / 2),
            'type': 'CH',
            'temp': BASE_TEMP,
            'energy': INITIAL_ENERGY * 100,
            'is_alive': True
        }

        # Uç Sensörler (Ormana rastgele dağıtılmış)
        for i in range(NUM_SENSORS):
            node_id = f"Sensor_{i}"
            pos_x = random.uniform(5, GRID_SIZE - 5)
            pos_y = random.uniform(5, GRID_SIZE - 5)
            self.nodes[node_id] = {
                'pos': (pos_x, pos_y),
                'type': 'Sensor',
                'temp': BASE_TEMP,
                'energy': INITIAL_ENERGY + random.uniform(-10, 10),  # Başlangıçta ufak pil farklılıkları
                'is_alive': True
            }

    def calculate_heatmap(self, time_now):
        """Zaman ve rüzgara göre ısı haritasını (Grid) hesaplar."""
        # Yangının etki alanı zamanla büyür
        spread = 5.0 + (time_now * FIRE_SPREAD_RATE)

        # Rüzgardan dolayı yangın merkezin kayması
        center_x = IGNITION_POINT[0] + (WIND_VECTOR[0] * time_now * 0.5)
        center_y = IGNITION_POINT[1] + (WIND_VECTOR[1] * time_now * 0.5)

        # Gauss tabanlı ısı dağılımı formülü (Mesafe karesi ters orantılı)
        dist_sq = ((self.X - center_x)**2) / (spread**2) + ((self.Y - center_y)**2) / (spread**2)

        # Grid üzerindeki sıcaklık hesabı
        temperature_grid = BASE_TEMP + (FIRE_TEMP_MAX * np.exp(-dist_sq))
        return temperature_grid

    def get_temperature_at(self, pos, heatmap):
        """Verilen (x, y) noktasındaki sıcaklığı ısımatrisinden okur."""
        x_idx = min(int(pos[0]), GRID_SIZE - 1)
        y_idx = min(int(pos[1]), GRID_SIZE - 1)
        return heatmap[y_idx, x_idx]  # Numpy meshgrid Y, X sıralı tutar

    def sensor_node(self, node_id):
        """Uç sensör simülasyonu."""
        while self.nodes[node_id]['is_alive']:
            node = self.nodes[node_id]

            # --- ÖLÇÜM AŞAMASI ---
            if node['energy'] < E_SENSE:
                node['is_alive'] = False  # Pili bitti, öldü
                break

            # Ölçüm (Sense) Enerjisi tüketimi
            node['energy'] -= E_SENSE
            self.total_energy_consumed_agg += E_SENSE
            self.total_energy_consumed_raw += E_SENSE

            # Anlık ısı haritasını alıp sensörünendi noktasındaki sıcaklığı okuma
            current_heatmap = self.calculate_heatmap(self.env.now)
            temp = self.get_temperature_at(node['pos'], current_heatmap)
            node['temp'] = temp

            # --- GÖNDERİM AŞAMASI (Sadece yangın > 45 derece ise ilet) ---
            if temp > 45.0:
                if node['energy'] >= E_TX:
                    node['energy'] -= E_TX
                    self.total_energy_consumed_agg += E_TX

                    # Küme Başına (CH) İletim yapıldı
                    self.ch_buffer.append({"node_id": node_id, "temp": temp})
                    self.total_packets_sent_raw += 1

                    # Eğer Ağda Aggregation (Birleştirme) olmasaydı:
                    # Direk Ana Merkeze gidecekti ve daha fazla enerji harcanacaktı.
                    self.total_energy_consumed_raw += E_TX
                else:
                    node['is_alive'] = False  # Gönderi yaparken pili bitti

            yield self.env.timeout(2)  # Sensör 2 saniyede bir dinler

    def cluster_head(self):
        """Küme Başı (CH) Gelen verileri birleştirip merkeze atar."""
        ch_node = self.nodes[self.cluster_head_id]

        while ch_node['is_alive']:
            yield self.env.timeout(2)  # CH 2 sn boyunca gelen paketleri dinler

            if len(self.ch_buffer) > 0:
                # Verileri Alma Maliyeti (Rx)
                rx_cost = len(self.ch_buffer) * E_RX
                ch_node['energy'] -= rx_cost
                self.total_energy_consumed_agg += rx_cost

                # Verileri İşleme/Birleştirme Maliyeti (Agg)
                ch_node['energy'] -= E_AGG
                self.total_energy_consumed_agg += E_AGG

                # Tek paket iletimi (Tx)
                if ch_node['energy'] >= E_TX:
                    ch_node['energy'] -= E_TX
                    self.total_energy_consumed_agg += E_TX

                    self.total_packets_sent_agg += 1

                self.ch_buffer = []

    def monitor(self):
        """Ağın anlık durumunu her saniye kaydeder (Animasyon İçin)."""
        while True:
            current_heatmap = self.calculate_heatmap(self.env.now)
            current_state = copy.deepcopy(self.nodes)

            alive_nodes = sum(1 for n in self.nodes.values() if n['type'] == 'Sensor' and n['is_alive'])

            stats = {
                'alive': alive_nodes,
                'raw_pkts': self.total_packets_sent_raw,
                'agg_pkts': self.total_packets_sent_agg,
                'agg_egy': self.total_energy_consumed_agg,
                'raw_egy': self.total_energy_consumed_raw
            }

            self.history.append({
                'time': self.env.now,
                'state': current_state,
                'heatmap': current_heatmap,
                'stats': stats
            })
            yield self.env.timeout(1)


def run_advanced_simulation():
    print("--- GELİŞMİŞ WSN SİMÜLASYONU BAŞLIYOR ---")
    print(f"Sensör Sayısı: {NUM_SENSORS}")
    print(f"Simülasyon Süresi: {SIMULATION_TIME} saniye")

    env = simpy.Environment()
    sim = WSN_Simulation(env)

    sim.init_network()

    for i in range(NUM_SENSORS):
        env.process(sim.sensor_node(node_id=f"Sensor_{i}"))

    env.process(sim.cluster_head())
    env.process(sim.monitor())

    env.run(until=SIMULATION_TIME)

    print("\n--- TRAFİK VE ENERJİ ANALİZİ ---")
    print(f"Veri Birleştirme (Aggregation) OLMASAYDI İletilecek Paket: {sim.total_packets_sent_raw}")
    print(f"Veri Birleştirme İLE Merkeze İletilen Paket: {sim.total_packets_sent_agg}")

    if sim.total_packets_sent_raw > 0:
        ratio = ((sim.total_packets_sent_raw - sim.total_packets_sent_agg) / sim.total_packets_sent_raw) * 100
        print(f"Ağ Trafiği Azalma Oranı: % {ratio:.2f}")

    print(f"\nBirleştirme OLMASAYDI Ağın Enerji Harcaması: {sim.total_energy_consumed_raw:.2f} mJ")
    print(f"Birleştirme İLE Ağın Enerji Harcaması: {sim.total_energy_consumed_agg:.2f} mJ")

    return sim.history


def create_animation(history):
    print("\n--- GÖRSELLEŞTİRME (ANİMASYON) HAZIRLANIYOR... ---")
    fig, (ax_main, ax_text) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [3, 1]})
    fig.canvas.manager.set_window_title('Gelişmiş Orman Yangını Data Aggregation Simülasyonu')

    # Text paneli (Dashboard)
    ax_text.axis('off')
    info_text = ax_text.text(0.0, 0.5, '', fontsize=13, va='center', ha='left', family='monospace')

    # Isı Haritası X,Y Matrisi
    x_grid = np.linspace(0, GRID_SIZE, GRID_SIZE)
    y_grid = np.linspace(0, GRID_SIZE, GRID_SIZE)
    X, Y = np.meshgrid(x_grid, y_grid)

    def update(frame_idx):
        ax_main.clear()

        record = history[frame_idx]
        t = record['time']
        state = record['state']
        heatmap = record['heatmap']
        stats = record['stats']

        # 1. Isı Haritasını Renkli Çiz (Mavi -> Kırmızı)
        contour_opts = {'levels': np.linspace(20, 800, 30), 'cmap': 'YlOrRd', 'alpha': 0.8}
        ax_main.contourf(X, Y, heatmap, **contour_opts)

        # 2. Sensörleri ve CH'i Çiz
        alive_xs, alive_ys, alive_colors = [], [], []
        dead_xs, dead_ys = [], []
        ch_x, ch_y = None, None

        for _, n_info in state.items():
            if n_info['type'] == 'CH':
                ch_x, ch_y = n_info['pos']
            else:
                if n_info['is_alive']:
                    alive_xs.append(n_info['pos'][0])
                    alive_ys.append(n_info['pos'][1])
                    # Sıcaklık > 45 ise farklı renk göster
                    if n_info['temp'] > 45:
                        alive_colors.append('#00FF00')  # Veri İleten Sensör
                    else:
                        alive_colors.append('#008000')  # Normal Sensör
                else:
                    dead_xs.append(n_info['pos'][0])
                    dead_ys.append(n_info['pos'][1])

        # Grafik noktalarını bas
        if alive_xs:
            ax_main.scatter(alive_xs, alive_ys, c=alive_colors, s=60, edgecolors='black', label='Canlı Sensörler')
        if dead_xs:
            ax_main.scatter(dead_xs, dead_ys, c='black', marker='X', s=70, label='Ölü (Pili Biten)')
        if ch_x is not None:
            ax_main.scatter([ch_x], [ch_y], c='blue', s=250, marker='*', edgecolors='white', label='Küme Başı (CH)')

        ax_main.set_xlim(0, GRID_SIZE)
        ax_main.set_ylim(0, GRID_SIZE)
        ax_main.set_title(f"Ağ İçi Veri Birleştirme (Kuzeydoğu Rüzgarlı Yangın Modeli)\nZaman: {t} saniye", fontweight='bold', fontsize=14)
        if frame_idx == 0:
            ax_main.legend(loc='upper left', fontsize=10)

        # 3. İstatistik Kurulumu (Gösterge Paneli)
        ratio = 0
        if stats['raw_pkts'] > 0:
            ratio = ((stats['raw_pkts'] - stats['agg_pkts']) / stats['raw_pkts']) * 100

        egy_savings = 0
        if stats['raw_egy'] > 0:
            egy_savings = ((stats['raw_egy'] - stats['agg_egy']) / stats['raw_egy']) * 100

        dashboard_str = (
            f"DURUM RAPORU\n"
            f"{'=' * 25}\n"
            f"Zaman        : {t} / {SIMULATION_TIME} sn\n\n"
            f"Sensör Durumu\n"
            f"{'-' * 25}\n"
            f"Toplam       : {NUM_SENSORS}\n"
            f"Canlı Sensör : {stats['alive']}\n"
            f"Ölü (Kayıp)  : {NUM_SENSORS - stats['alive']}\n\n"
            f"Ağ Trafiği\n"
            f"{'-' * 25}\n"
            f"Ham Pkt (Tx) : {stats['raw_pkts']}\n"
            f"Agg Pkt (Tx) : {stats['agg_pkts']}\n"
            f"Trafik Düşüş : % {ratio:.1f}\n\n"
            f"Enerji Analizi (mJ)\n"
            f"{'-' * 25}\n"
            f"Ham Tüketim  : {stats['raw_egy']:.1f}\n"
            f"Agg Tüketim  : {stats['agg_egy']:.1f}\n"
            f"Enerji Tas.  : % {egy_savings:.1f}\n"
        )
        info_text.set_text(dashboard_str)
        fig.tight_layout()

    # Animasyonu oluştur
    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=300, repeat=False)
    plt.show()
    return ani


if __name__ == "__main__":
    history_data = run_advanced_simulation()
    create_animation(history_data)
