"""
Orman Yangını WSN Simülasyonu - Pygame Görselleştirme
------------------------------------------------------
SimPy ile çalışan Data Aggregation simülasyonunu
Pygame ile gerçek zamanlı, görsel açıdan zengin bir
arayüzde gösterir.

Özellikler:
  - Çoklu Küme Başı (CH) ile gerçekçi WSN topolojisi
  - Ateş parçacık sistemi (particle system)
  - Sıcaklığa göre renk geçişli zemin ısı haritası
  - Sensörlere "ping" halkası iletim animasyonu
  - Bağlantı ışık çizgileri (sensor -> CH)
  - Sağ panelde kaydırılabilir (scroll) istatistik göstergesi
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field


import numpy as np
import pygame
import simpy

# ── SIMÜLASYON PARAMETRELERİ ─────────────────────────────────────────────────
GRID_SIZE: int = 100
NUM_SENSORS: int = 150
NUM_CHS: int = 4                     # Küme Başı sayısı
SIMULATION_TIME: int = 60

INITIAL_ENERGY: float = 100.0
E_SENSE: float = 0.5
E_TX: float = 2.0
E_RX: float = 1.0
E_AGG: float = 1.5

IGNITION_POINT: tuple[float, float] = (30.0, 30.0)
WIND_VECTOR: tuple[float, float] = (1.5, 0.8)
FIRE_SPREAD_RATE: float = 1.2
BASE_TEMP: float = 20.0
FIRE_TEMP_MAX: float = 800.0
FIRE_THRESHOLD: float = 45.0

# ── PYGAME EKRAN PARAMETRELERİ ────────────────────────────────────────────────
WIN_W: int = 1600
WIN_H: int = 850
SIM_W: int = 1000
PANEL_W: int = WIN_W - SIM_W
FPS: int = 60
CELL: float = SIM_W / GRID_SIZE

# ── RENKLER ───────────────────────────────────────────────────────────────────
Color = tuple[int, int, int]

C_BG: Color = (10, 10, 20)
C_PANEL: Color = (20, 24, 40)
C_TITLE: Color = (255, 200, 60)
C_TEXT: Color = (210, 215, 230)
C_ALIVE: Color = (40, 230, 100)
C_FIRE_NODE: Color = (255, 120, 30)
C_DEAD: Color = (80, 80, 80)
C_CH: Color = (60, 140, 255)
C_LINK: Color = (255, 240, 100)
C_PING: Color = (100, 220, 255)

# Her CH'ye farklı renk
CH_COLORS: list[Color] = [
    (60, 140, 255),    # Mavi
    (200, 80, 255),    # Mor
    (255, 160, 40),    # Turuncu
    (40, 220, 200),    # Turkuaz
]


# ── VERİ YAPILARI ─────────────────────────────────────────────────────────────
@dataclass
class Node:
    """Ağdaki her düğümü temsil eden tip-güvenli veri sınıfı."""
    node_id: str
    pos: tuple[float, float]
    node_type: str          # "CH" veya "Sensor"
    cluster_ch: str = ""    # Bu sensörün bağlı olduğu CH id
    temp: float = BASE_TEMP
    energy: float = INITIAL_ENERGY
    is_alive: bool = True


@dataclass
class FrameStats:
    """Bir simülasyon anındaki istatistikler."""
    alive: int
    raw_pkts: int
    agg_pkts: int
    agg_egy: float
    raw_egy: float


@dataclass
class HistoryRecord:
    """Bir simülasyon anının tam anlık görüntüsü."""
    time: int
    nodes: dict[str, Node]
    heatmap: np.ndarray
    stats: FrameStats


# ── YARDIMCI FONKSİYONLAR ────────────────────────────────────────────────────
def lerp_color(c1: Color, c2: Color, t: float) -> Color:
    """İki renk arasında doğrusal ara geçiş (t: 0.0-1.0)."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def sim_to_screen(x: float, y: float) -> tuple[int, int]:
    """Simülasyon koordinatını (0-100) piksel koordinatına çevirir."""
    return int(x * CELL), int(WIN_H - y * (WIN_H / GRID_SIZE))


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """İki nokta arasındaki Öklid mesafesini hesaplar."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ── PARTİKÜL SİSTEMİ ─────────────────────────────────────────────────────────
@dataclass
class Particle:
    x: float
    y: float
    vx: float = field(init=False)
    vy: float = field(init=False)
    life: float = field(init=False)
    max_life: float = field(init=False)
    radius: int = field(init=False)
    color_hot: Color = field(init=False)

    def __post_init__(self) -> None:
        angle: float = random.uniform(60.0, 120.0)
        speed: float = random.uniform(0.5, 2.0)
        self.vx = math.cos(math.radians(angle)) * speed + WIND_VECTOR[0] * 0.08
        self.vy = -math.sin(math.radians(angle)) * speed - WIND_VECTOR[1] * 0.04
        self.life = random.uniform(0.5, 1.0)
        self.max_life = self.life
        self.radius = random.randint(2, 5)
        self.color_hot = random.choice([
            (255, 80, 0), (255, 160, 0), (255, 230, 60)
        ])

    def update(self, dt: float) -> None:
        self.x += self.vx
        self.y += self.vy
        self.vy -= 0.03
        self.life -= dt

    def draw(self, surface: pygame.Surface) -> None:
        ratio: float = self.life / self.max_life
        color: Color = lerp_color((50, 50, 50), self.color_hot, ratio)
        alpha_surf = pygame.Surface(
            (self.radius * 2, self.radius * 2), pygame.SRCALPHA
        )
        pygame.draw.circle(
            alpha_surf,
            (*color, int(ratio * 200)),
            (self.radius, self.radius),
            self.radius,
        )
        surface.blit(
            alpha_surf,
            (int(self.x - self.radius), int(self.y - self.radius)),
        )

    @property
    def alive(self) -> bool:
        return self.life > 0


# ── PING HALKASI ──────────────────────────────────────────────────────────────
@dataclass
class PingRing:
    x: int
    y: int
    color: Color
    radius: float = 6.0
    max_radius: float = 28.0
    life: float = 1.0

    def update(self, dt: float) -> None:
        self.life -= dt * 1.8
        self.radius = 6.0 + (1.0 - self.life) * self.max_radius

    def draw(self, surface: pygame.Surface) -> None:
        if self.life <= 0:
            return
        alpha: int = int(self.life * 200)
        r: int = int(self.radius)
        ring_surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(
            ring_surf, (*self.color, alpha), (r + 2, r + 2), r, 2
        )
        surface.blit(ring_surf, (self.x - r - 2, self.y - r - 2))

    @property
    def alive(self) -> bool:
        return self.life > 0


# ── BAĞLANTI IŞIĞI ────────────────────────────────────────────────────────────
@dataclass
class LinkFlash:
    p1: tuple[int, int]
    p2: tuple[int, int]
    life: float = 1.0

    def update(self, dt: float) -> None:
        self.life -= dt * 2.5

    def draw(self, surface: pygame.Surface) -> None:
        if self.life <= 0:
            return
        alpha: int = int(self.life * 180)
        width: int = max(1, int(self.life * 2))
        link_surf = pygame.Surface((SIM_W, WIN_H), pygame.SRCALPHA)
        pygame.draw.line(
            link_surf, (*C_LINK, alpha), self.p1, self.p2, width
        )
        surface.blit(link_surf, (0, 0))

    @property
    def alive(self) -> bool:
        return self.life > 0


# ── WSN SİMÜLASYON MODELİ ────────────────────────────────────────────────────
class WSN_Simulation:
    def __init__(self, env: simpy.Environment) -> None:
        self.env = env
        self.total_packets_sent_raw: int = 0
        self.total_packets_sent_agg: int = 0
        self.total_energy_consumed_raw: float = 0.0
        self.total_energy_consumed_agg: float = 0.0
        self.ch_buffers: dict[str, list[dict[str, object]]] = {}
        self.nodes: dict[str, Node] = {}
        self.history: list[HistoryRecord] = []
        self.ch_ids: list[str] = []

        x = np.linspace(0, GRID_SIZE, GRID_SIZE)
        y = np.linspace(0, GRID_SIZE, GRID_SIZE)
        self.X: np.ndarray
        self.Y: np.ndarray
        self.X, self.Y = np.meshgrid(x, y)

    def init_network(self) -> None:
        """Ağı, çoklu CH'leri ve sensörleri başlat."""
        # CH'leri grid üzerine dengeli dağıt
        ch_positions: list[tuple[float, float]] = [
            (25.0, 75.0),   # Sol üst
            (75.0, 75.0),   # Sağ üst
            (25.0, 25.0),   # Sol alt
            (75.0, 25.0),   # Sağ alt
        ]
        for i in range(NUM_CHS):
            ch_id = f"CH_{i}"
            self.ch_ids.append(ch_id)
            self.ch_buffers[ch_id] = []
            self.nodes[ch_id] = Node(
                node_id=ch_id,
                pos=ch_positions[i],
                node_type="CH",
                cluster_ch=ch_id,
                temp=BASE_TEMP,
                energy=INITIAL_ENERGY * 100.0,
                is_alive=True,
            )

        # Sensörleri grid tabanlı hafif kaydırarak daha homojen (orman gibi) yerleştir
        import math
        cols: int = int(math.ceil(math.sqrt(NUM_SENSORS)))
        rows: int = int(math.ceil(NUM_SENSORS / cols))
        cell_w: float = (GRID_SIZE - 6.0) / cols
        cell_h: float = (GRID_SIZE - 6.0) / rows

        for i in range(NUM_SENSORS):
            nid = f"Sensor_{i}"
            r_idx: int = i // cols
            c_idx: int = i % cols
            
            # Merkeze yerleştirip rastgele (jitter) yönlerde hafifçe kaydır
            cx: float = 3.0 + c_idx * cell_w + cell_w / 2.0
            cy: float = 3.0 + r_idx * cell_h + cell_h / 2.0
            jx: float = random.uniform(-cell_w * 0.4, cell_w * 0.4)
            jy: float = random.uniform(-cell_h * 0.4, cell_h * 0.4)
            pos = (cx + jx, cy + jy)

            # En yakın CH'yi bul
            nearest_ch = min(
                self.ch_ids,
                key=lambda cid: distance(pos, self.nodes[cid].pos),
            )
            self.nodes[nid] = Node(
                node_id=nid,
                pos=pos,
                node_type="Sensor",
                cluster_ch=nearest_ch,
                temp=BASE_TEMP,
                energy=INITIAL_ENERGY + random.uniform(-10.0, 10.0),
                is_alive=True,
            )

    def calculate_heatmap(self, time_now: float) -> np.ndarray:
        """Zaman ve rüzgara göre ısı haritasını hesaplar."""
        spread: float = 5.0 + time_now * FIRE_SPREAD_RATE
        cx: float = IGNITION_POINT[0] + WIND_VECTOR[0] * time_now * 0.5
        cy: float = IGNITION_POINT[1] + WIND_VECTOR[1] * time_now * 0.5
        dist_sq: np.ndarray = (
            ((self.X - cx) ** 2) / (spread ** 2)
            + ((self.Y - cy) ** 2) / (spread ** 2)
        )
        return BASE_TEMP + FIRE_TEMP_MAX * np.exp(-dist_sq)

    def get_temperature_at(
        self, pos: tuple[float, float], heatmap: np.ndarray
    ) -> float:
        """Verilen (x, y) noktasındaki sıcaklığı haritadan okur."""
        xi: int = min(int(pos[0]), GRID_SIZE - 1)
        yi: int = min(int(pos[1]), GRID_SIZE - 1)
        return float(heatmap[yi, xi])

    def sensor_node(self, node_id: str):  # type: ignore[return]
        """Uç sensör simülasyonu (SimPy generator)."""
        while self.nodes[node_id].is_alive:
            node: Node = self.nodes[node_id]
            
            # Sadece yaşamak/ölçüm yapmak için harcanan temel enerji
            if node.energy < E_SENSE:
                node.is_alive = False
                node.energy = 0
                break
                
            node.energy -= E_SENSE
            self.total_energy_consumed_agg += E_SENSE
            self.total_energy_consumed_raw += E_SENSE

            heatmap: np.ndarray = self.calculate_heatmap(float(self.env.now))
            temp: float = self.get_temperature_at(node.pos, heatmap)
            node.temp = temp

            # Yangın durumunda
            if temp > FIRE_THRESHOLD:
                # ── Yangın Hasar (Fire Damage) Mekanizması ──
                # Sıcaklık arttıkça ağacın canı (enerjisi) yanıp tükenir.
                # Daha yavaş ve dengeli yanma için hasar çarpanı 0.02'ye düşürüldü.
                # Örn: TEMP 400 ise, (400 - 45) * 0.02 = ~7 enerji hasarı (15 sn'de ölür)
                fire_damage: float = (temp - FIRE_THRESHOLD) * 0.02
                node.energy -= fire_damage
                
                if node.energy <= 0.0:
                    node.energy = 0.0
                    node.is_alive = False
                    break # Ağaç tamamen yandı ve öldü/kurudu

                # Eğer hala hayattaysa veriyi CH'ye gönder
                if node.energy >= E_TX:
                    node.energy -= E_TX
                    self.total_energy_consumed_agg += E_TX
                    self.total_energy_consumed_raw += E_TX
                    # Kendi CH'sine gönder
                    self.ch_buffers[node.cluster_ch].append(
                        {"node_id": node_id, "temp": temp}
                    )
                    self.total_packets_sent_raw += 1
                else:
                    node.is_alive = False
                    node.energy = 0.0
                    break

            yield self.env.timeout(2)

    def cluster_head_process(self, ch_id: str):  # type: ignore[return]
        """Küme Başı işlemi (SimPy generator). Her CH bağımsız çalışır."""
        ch: Node = self.nodes[ch_id]
        while ch.is_alive:
            yield self.env.timeout(2)
            buf = self.ch_buffers[ch_id]
            if buf:
                rx_cost: float = len(buf) * E_RX
                ch.energy -= rx_cost
                self.total_energy_consumed_agg += rx_cost
                ch.energy -= E_AGG
                self.total_energy_consumed_agg += E_AGG
                if ch.energy >= E_TX:
                    ch.energy -= E_TX
                    self.total_energy_consumed_agg += E_TX
                    self.total_packets_sent_agg += 1
                self.ch_buffers[ch_id] = []

    def monitor(self):  # type: ignore[return]
        """Her saniyede ağın anlık durumunu kaydeder."""
        while True:
            heatmap: np.ndarray = self.calculate_heatmap(float(self.env.now))
            alive: int = sum(
                1 for n in self.nodes.values()
                if n.node_type == "Sensor" and n.is_alive
            )
            self.history.append(
                HistoryRecord(
                    time=int(self.env.now),
                    nodes=copy.deepcopy(self.nodes),
                    heatmap=heatmap,
                    stats=FrameStats(
                        alive=alive,
                        raw_pkts=self.total_packets_sent_raw,
                        agg_pkts=self.total_packets_sent_agg,
                        agg_egy=self.total_energy_consumed_agg,
                        raw_egy=self.total_energy_consumed_raw,
                    ),
                )
            )
            yield self.env.timeout(1)


# ── HEATMAP SURFACE ───────────────────────────────────────────────────────────
# Vektörel Optimizasyon: NumPy (surfarray) kullanımı ile CPU hızlandırıldı.
def build_heatmap_surface(heatmap_array: np.ndarray) -> pygame.Surface:
    """Numpy ısı haritasını vektörel (Pygame surfarray) yöntemle çevirir."""
    # ratio: 0.0 (SOĞUK/BASE) ile 1.0 (HOT) arası
    ratio = np.clip((heatmap_array - BASE_TEMP) / (FIRE_TEMP_MAX - BASE_TEMP), 0.0, 1.0)
    
    # Renk tanımları (R, G, B)
    cold = np.array([0, 30, 80], dtype=np.float32)
    warm = np.array([200, 80, 0], dtype=np.float32)
    hot = np.array([255, 240, 60], dtype=np.float32)
    
    # Maske (ratio < 0.5 ve >= 0.5 durumları)
    mask_low = ratio < 0.5
    mask_high = ~mask_low
    
    # Yeni oran matrisleri
    t_low = ratio * 2.0
    t_high = (ratio - 0.5) * 2.0
    
    # Çıkış matrisi (RGB kanalları, shape: 100x100x3)
    rgb = np.empty((*ratio.shape, 3), dtype=np.float32)
    
    # low durum (Cold -> Warm)
    t_low_expand = np.expand_dims(t_low[mask_low], axis=-1)
    rgb[mask_low] = cold + (warm - cold) * t_low_expand
    
    # high durum (Warm -> Hot)
    t_high_expand = np.expand_dims(t_high[mask_high], axis=-1)
    rgb[mask_high] = warm + (hot - warm) * t_high_expand
    
    rgb_uint8 = rgb.astype(np.uint8)
    
    # Pygame 좌표ları (x, y) = (col, row), fakat Numpy matrix [row, col]
    # np.transpose yardımıyla veya surfarray kurgusuyla ekranla hizalamak
    # Sütunlar X, satırlar Y. Origin Pygame'de üst sol=0,0
    # Fakat hesaplamada Y yukarı doğru olabilir: 
    # Y-eksenini tersine çeviriyoruz (GRID_SIZE - 1 - gy mantığı)
    rgb_uint8 = np.flip(rgb_uint8, axis=0) # Satırları (Y ekseni) dikey çevir
    
    # surfarray.make_surface (Shape: [x/width, y/height, rgb]) istiyor
    # Bizdeki `rgb_uint8` (Y, X, C) formatında. Bunu (X, Y, C) yapalım:
    rgb_pygame = np.transpose(rgb_uint8, (1, 0, 2))
    
    surf = pygame.surfarray.make_surface(rgb_pygame)
    return pygame.transform.scale(surf, (SIM_W, WIN_H))


# ── YAZI ÇİZ ─────────────────────────────────────────────────────────────────
def draw_text(
    surface: pygame.Surface,
    text: str,
    x: int,
    y: int,
    font: pygame.font.Font,
    color: Color = C_TEXT,
) -> None:
    surface.blit(font.render(text, True, color), (x, y))


# ── SAĞ PANEL (kaydırılabilir) ────────────────────────────────────────────────
def draw_panel(
    surface: pygame.Surface,
    stats: FrameStats,
    frame_no: int,
    total_frames: int,
    nodes: dict[str, Node],
    font_big: pygame.font.Font,
    font_med: pygame.font.Font,
    font_sm: pygame.font.Font,
    scroll_y: int,
) -> int:
    """Sağ paneli çizer ve toplam panel yüksekliğini döndürür."""
    # Panel arka planını çiz
    panel_rect = pygame.Rect(SIM_W, 0, PANEL_W, WIN_H)
    pygame.draw.rect(surface, C_PANEL, panel_rect)
    pygame.draw.line(surface, C_TITLE, (SIM_W, 0), (SIM_W, WIN_H), 2)

    # Clip bölgesi – panel dışına taşmayı önle
    clip_rect = pygame.Rect(SIM_W + 2, 2, PANEL_W - 4, WIN_H - 4)
    surface.set_clip(clip_rect)

    px: int = SIM_W + 14
    pw: int = PANEL_W - 28
    y: int = 18 - scroll_y

    draw_text(surface, "DURUM RAPORU", px, y, font_big, C_TITLE)
    y += 28
    pygame.draw.line(surface, C_TITLE, (SIM_W + 8, y), (WIN_W - 8, y), 1)
    y += 10

    # Zaman ilerleme barı
    prog: float = frame_no / max(1, total_frames)
    draw_text(
        surface,
        f"Zaman : {frame_no} / {total_frames} sn",
        px, y, font_med,
    )
    y += 20
    pygame.draw.rect(surface, (50, 50, 80), (px, y, pw, 10), border_radius=5)
    pygame.draw.rect(
        surface, C_TITLE, (px, y, int(pw * prog), 10), border_radius=5
    )
    y += 18
    pygame.draw.line(surface, (60, 65, 90), (SIM_W + 8, y), (WIN_W - 8, y), 1)
    y += 8

    # Sensör durumu
    alive: int = stats.alive
    dead: int = NUM_SENSORS - alive
    draw_text(surface, "SENSÖRLER", px, y, font_med, C_TITLE)
    y += 18
    draw_text(surface, f"Toplam    : {NUM_SENSORS}", px, y, font_sm)
    y += 15
    draw_text(surface, f"Canlı     : {alive}", px, y, font_sm, C_ALIVE)
    y += 15
    draw_text(surface, f"Ölü       : {dead}", px, y, font_sm, (220, 70, 70))
    y += 20
    pygame.draw.rect(surface, (50, 50, 80), (px, y, pw, 10), border_radius=4)
    alive_ratio: float = alive / NUM_SENSORS
    pygame.draw.rect(
        surface, C_ALIVE, (px, y, int(pw * alive_ratio), 10), border_radius=4
    )
    y += 18
    pygame.draw.line(surface, (60, 65, 90), (SIM_W + 8, y), (WIN_W - 8, y), 1)
    y += 8

    # ── Küme Başları Durumu ───────────────────────────────────────────────
    draw_text(surface, "KÜME BAŞLARI (CH)", px, y, font_med, C_TITLE)
    y += 18
    for i, ch_id in enumerate(
        sorted(nid for nid, n in nodes.items() if n.node_type == "CH")
    ):
        ch_node = nodes[ch_id]
        color = CH_COLORS[i % len(CH_COLORS)]
        # CH'ye bağlı sensör sayısı
        cluster_count = sum(
            1 for n in nodes.values()
            if n.node_type == "Sensor" and n.cluster_ch == ch_id
        )
        cluster_alive = sum(
            1 for n in nodes.values()
            if n.node_type == "Sensor" and n.cluster_ch == ch_id and n.is_alive
        )
        status = "●" if ch_node.is_alive else "✕"
        draw_text(
            surface,
            f"{status} {ch_id}: {cluster_alive}/{cluster_count} sensör",
            px, y, font_sm, color,
        )
        y += 14
        # Enerji barı
        ch_ratio = max(0.0, ch_node.energy / (INITIAL_ENERGY * 100.0))
        bar_w = pw - 20
        pygame.draw.rect(
            surface, (50, 50, 80), (px + 10, y, bar_w, 6), border_radius=3
        )
        bar_clr = lerp_color((220, 60, 60), color, ch_ratio)
        pygame.draw.rect(
            surface, bar_clr,
            (px + 10, y, int(bar_w * ch_ratio), 6), border_radius=3,
        )
        y += 12
    y += 4
    pygame.draw.line(surface, (60, 65, 90), (SIM_W + 8, y), (WIN_W - 8, y), 1)
    y += 8

    # Ağ trafiği
    draw_text(surface, "AĞ TRAFİĞİ", px, y, font_med, C_TITLE)
    y += 18
    rp: int = stats.raw_pkts
    ap: int = stats.agg_pkts
    draw_text(surface, f"Ham Paket : {rp}", px, y, font_sm)
    y += 15
    draw_text(surface, f"Agg Paket : {ap}", px, y, font_sm, C_PING)
    y += 15
    t_ratio: float = (rp - ap) / rp * 100.0 if rp > 0 else 0.0
    draw_text(
        surface, f"Azalma    : %{t_ratio:.1f}", px, y, font_sm, C_TITLE
    )
    y += 20
    pygame.draw.line(surface, (60, 65, 90), (SIM_W + 8, y), (WIN_W - 8, y), 1)
    y += 8

    # Enerji
    draw_text(surface, "ENERJİ (mJ)", px, y, font_med, C_TITLE)
    y += 18
    re: float = stats.raw_egy
    ae: float = stats.agg_egy
    draw_text(surface, f"Ham Tük.  : {re:.1f}", px, y, font_sm)
    y += 15
    draw_text(surface, f"Agg Tük.  : {ae:.1f}", px, y, font_sm, C_PING)
    y += 15
    esav: float = (re - ae) / re * 100.0 if re > 0.0 else 0.0
    e_color: Color = C_ALIVE if esav >= 0.0 else (220, 70, 70)
    draw_text(surface, f"Tasarruf  : %{esav:.1f}", px, y, font_sm, e_color)
    y += 20
    pygame.draw.line(surface, (60, 65, 90), (SIM_W + 8, y), (WIN_W - 8, y), 1)
    y += 8

    # Rüzgar pusulası
    draw_text(surface, "RÜZGAR YÖNÜ", px, y, font_med, C_TITLE)
    y += 20
    compass_cx: int = SIM_W + PANEL_W // 2
    compass_cy: int = y + 26
    pygame.draw.circle(surface, (40, 44, 70), (compass_cx, compass_cy), 26, 1)
    vx: float = WIND_VECTOR[0]
    vy: float = WIND_VECTOR[1]
    mag: float = math.hypot(vx, vy)
    wnx: float = vx / mag
    wny: float = -vy / mag
    arrow_ex: int = int(compass_cx + wnx * 20)
    arrow_ey: int = int(compass_cy + wny * 20)
    pygame.draw.line(
        surface, (255, 120, 30),
        (compass_cx, compass_cy), (arrow_ex, arrow_ey), 3,
    )
    pygame.draw.circle(surface, (255, 120, 30), (arrow_ex, arrow_ey), 4)
    draw_text(surface, "K", compass_cx - 5, compass_cy - 40, font_sm, (150, 160, 180))
    draw_text(surface, "G", compass_cx - 5, compass_cy + 12, font_sm, (150, 160, 180))
    draw_text(surface, "B", compass_cx - 40, compass_cy - 6, font_sm, (150, 160, 180))
    draw_text(surface, "D", compass_cx + 30, compass_cy - 6, font_sm, (150, 160, 180))
    y = compass_cy + 40
    pygame.draw.line(surface, (60, 65, 90), (SIM_W + 8, y), (WIN_W - 8, y), 1)
    y += 8

    # ── LEGEND (Renk Açıklamaları) ──────────────────────────────────────────
    draw_text(surface, "SEMBOLLER", px, y, font_med, C_TITLE)
    y += 18
    legend_items: list[tuple[Color, str]] = [
        (C_ALIVE, "Sağlıklı Ağaç (Sensör)"),
        (C_FIRE_NODE, "Yanan Ağaç (Algılama)"),
        ((100, 90, 80), "Kurumuş/Ölü Ağaç"),
        (C_LINK, "Veri İletim Bağlantısı"),
        (C_PING, "Ping Halkası (İletim)"),
    ]
    for i, ch_id in enumerate(sorted(
        nid for nid, n in nodes.items() if n.node_type == "CH"
    )):
        legend_items.append((CH_COLORS[i % len(CH_COLORS)], f"{ch_id} Küme Başı"))

    for clr, label in legend_items:
        pygame.draw.rect(surface, clr, (px, y + 2, 12, 12), border_radius=2)
        draw_text(surface, label, px + 18, y, font_sm)
        y += 16
    y += 4
    pygame.draw.line(surface, (60, 65, 90), (SIM_W + 8, y), (WIN_W - 8, y), 1)
    y += 8

    # ── SİMÜLASYON AÇIKLAMASI ───────────────────────────────────────────────
    draw_text(surface, "SİMÜLASYON HAKKINDA", px, y, font_med, C_TITLE)
    y += 18
    desc_lines: list[str] = [
        "Bu simülasyon, orman yangını ortamında",
        "kablosuz sensör ağının (WSN) veri",
        "toplama (data aggregation) sürecini",
        "modellemektedir.",
        "",
        f"{NUM_CHS} adet Küme Başı (CH) ormana",
        "dağıtılmıştır. Her sensör en yakın",
        "CH'ye otomatik bağlanır ve sıcaklık",
        "verilerini bu CH'ye iletir.",
        "",
        "Her CH, kendi kümesindeki sensörlerden",
        "gelen verileri birleştirerek (aggregation)",
        "paket sayısını ve enerji tüketimini azaltır.",
        "",
        "Ham iletim ile aggregation yöntemi",
        "karşılaştırılarak verimlilik farkı",
        "gerçek zamanlı olarak gösterilir.",
    ]
    for line in desc_lines:
        draw_text(surface, line, px, y, font_sm, (170, 175, 190))
        y += 14

    # Clip'i sıfırla
    surface.set_clip(None)

    # Toplam içerik yüksekliğini döndür (scroll limiti için)
    total_content_h: int = y + scroll_y + 20
    return total_content_h


# ── ANA PYGAME DÖNGÜSÜ ────────────────────────────────────────────────────────
def run_pygame_visualization(history: list[HistoryRecord]) -> None:
    pygame.init()
    screen: pygame.Surface = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(
        "Orman Yangını WSN - Çoklu CH Data Aggregation Simülasyonu"
    )
    clock: pygame.time.Clock = pygame.time.Clock()

    font_big: pygame.font.Font = pygame.font.SysFont("consolas", 16, bold=True)
    font_med: pygame.font.Font = pygame.font.SysFont("consolas", 13, bold=True)
    font_sm: pygame.font.Font = pygame.font.SysFont("consolas", 12)

    particles: list[Particle] = []
    pings: list[PingRing] = []
    links: list[LinkFlash] = []

    # CH pozisyonlarını ekran koordinatına çevir
    ch_screen_pos: dict[str, tuple[int, int]] = {}
    for nid, n in history[0].nodes.items():
        if n.node_type == "CH":
            ch_screen_pos[nid] = sim_to_screen(*n.pos)

    frame_idx: int = 0
    total_frames: int = len(history)
    running: bool = True
    prev_senders: set[str] = set()

    frame_ticker: int = 0
    # FPS artacagi için saniyedeki kare değişimi oranını koru (örneğin daha yavaş izlemek için saniyede az simülasyon karesi gösterilir)
    ticks_per_frame: int = max(1, FPS // 2)

    is_paused: bool = False
    pause_btn_rect: pygame.Rect = pygame.Rect(WIN_W - 140, 15, 120, 32)

    # Panel scroll
    panel_scroll_y: int = 0
    panel_content_h: int = WIN_H
    scroll_speed: int = 20

    while running:
        dt: float = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    is_paused = not is_paused
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pause_btn_rect.collidepoint(event.pos):
                    is_paused = not is_paused
            # Fare tekerleği ile panel scroll
            if event.type == pygame.MOUSEWHEEL:
                mx, _ = pygame.mouse.get_pos()
                if mx > SIM_W:  # sadece panelde scroll
                    panel_scroll_y -= event.y * scroll_speed
                    max_scroll = max(0, panel_content_h - WIN_H)
                    panel_scroll_y = max(0, min(panel_scroll_y, max_scroll))

        if not is_paused:
            frame_ticker += 1
            if frame_ticker >= ticks_per_frame and frame_idx < total_frames - 1:
                frame_ticker = 0
                frame_idx += 1

        record: HistoryRecord = history[frame_idx]
        nodes: dict[str, Node] = record.nodes
        heatmap: np.ndarray = record.heatmap
        stats: FrameStats = record.stats

        if not is_paused:
            # Parçacık üretimi (yangın bölgelerinden)
            for n in nodes.values():
                if n.node_type == "Sensor" and n.temp > FIRE_THRESHOLD:
                    if random.random() < 0.20:
                        sx: int
                        sy: int
                        sx, sy = sim_to_screen(*n.pos)
                        particles.append(Particle(float(sx), float(sy)))

            cx_fire: float = (
                IGNITION_POINT[0] + WIND_VECTOR[0] * record.time * 0.5
            )
            cy_fire: float = (
                IGNITION_POINT[1] + WIND_VECTOR[1] * record.time * 0.5
            )
            fpx: int
            fpy: int
            fpx, fpy = sim_to_screen(cx_fire, cy_fire)
            for _ in range(4):
                particles.append(
                    Particle(
                        float(fpx + random.randint(-8, 8)),
                        float(fpy + random.randint(-8, 8)),
                    )
                )

            # Ping ve link efektleri
            cur_senders: set[str] = set()
            for n_id, n in nodes.items():
                if (
                    n.node_type == "Sensor"
                    and n.temp > FIRE_THRESHOLD
                    and n.is_alive
                ):
                    cur_senders.add(n_id)
                    if n_id not in prev_senders:
                        ns_x: int
                        ns_y: int
                        ns_x, ns_y = sim_to_screen(*n.pos)
                        pings.append(PingRing(ns_x, ns_y, C_PING))
                        # Kendi CH'sine link
                        ch_target = n.cluster_ch
                        if ch_target in ch_screen_pos:
                            links.append(
                                LinkFlash(
                                    (ns_x, ns_y), ch_screen_pos[ch_target]
                                )
                            )
            prev_senders = cur_senders

            for p in particles:
                p.update(dt)
            for r in pings:
                r.update(dt)
            for lk in links:
                lk.update(dt)
            particles = [p for p in particles if p.alive]
            pings = [r for r in pings if r.alive]
            links = [lk for lk in links if lk.alive]

        # ── Çizim ─────────────────────────────────────────────────────────────
        screen.fill(C_BG)
        screen.blit(build_heatmap_surface(heatmap), (0, 0))

        for lk in links:
            lk.draw(screen)
        for p in particles:
            p.draw(screen)

        for n in nodes.values():
            sx2: int
            sy2: int
            sx2, sy2 = sim_to_screen(*n.pos)
            if n.node_type == "CH":
                # CH düğümü – renkli daire + CH yazısı
                ch_idx = int(n.node_id.split("_")[1]) if "_" in n.node_id else 0
                ch_color = CH_COLORS[ch_idx % len(CH_COLORS)]
                pygame.draw.circle(screen, ch_color, (sx2, sy2), 14)
                pygame.draw.circle(
                    screen, (220, 230, 255), (sx2, sy2), 14, 2
                )
                screen.blit(
                    font_sm.render("CH", True, (255, 255, 255)),
                    (sx2 - 9, sy2 - 7),
                )
            elif n.is_alive:
                # Ağaç şekli: üçgen taç + dikdörtgen gövde
                is_burning: bool = n.temp > FIRE_THRESHOLD
                if is_burning:
                    trunk_clr: Color = (100, 50, 20)
                    crown_clr: Color = C_FIRE_NODE
                else:
                    trunk_clr = (110, 70, 30)
                    crown_clr = C_ALIVE
                # Gövde (kahverengi dikdörtgen)
                pygame.draw.rect(
                    screen, trunk_clr,
                    (sx2 - 2, sy2, 4, 7),
                )
                # Taç (üçgen)
                crown_points = [
                    (sx2, sy2 - 9),
                    (sx2 - 6, sy2 + 2),
                    (sx2 + 6, sy2 + 2),
                ]
                pygame.draw.polygon(screen, crown_clr, crown_points)
                pygame.draw.polygon(
                    screen, (255, 255, 255), crown_points, 1
                )
                # Enerji barı (ağacın üstünde)
                energy_ratio: float = max(
                    0.0, n.energy / INITIAL_ENERGY
                )
                bar_w: int = 12
                pygame.draw.rect(
                    screen, (60, 60, 60),
                    (sx2 - 6, sy2 - 16, bar_w, 3),
                )
                bar_color: Color = lerp_color(
                    (220, 60, 60), C_ALIVE, energy_ratio
                )
                pygame.draw.rect(
                    screen, bar_color,
                    (sx2 - 6, sy2 - 16, int(bar_w * energy_ratio), 3),
                )
            else:
                # Ölü düğüm – Kurumuş / Ölü Ağaç Görseli
                trunk_clr = (80, 75, 70)   # Gri/soluk kahverengi gövde
                branch_clr = (100, 90, 80) # Kuru dallar
                # Gövde
                pygame.draw.rect(
                    screen, trunk_clr,
                    (sx2 - 2, sy2, 4, 7),
                )
                # Kuru dallar (sadece çizgiler, yaprak yok)
                pygame.draw.line(screen, branch_clr, (sx2, sy2), (sx2 - 4, sy2 - 6), 2)
                pygame.draw.line(screen, branch_clr, (sx2, sy2), (sx2 + 4, sy2 - 6), 2)
                pygame.draw.line(screen, branch_clr, (sx2, sy2 - 2), (sx2, sy2 - 10), 2)

        for r in pings:
            r.draw(screen)

        # Simülasyon alanı çerçevesi
        pygame.draw.rect(screen, (60, 65, 90), (0, 0, SIM_W, WIN_H), 1)

        # Panel çiz (scroll destekli)
        panel_content_h = draw_panel(
            screen, stats, record.time, SIMULATION_TIME,
            nodes, font_big, font_med, font_sm, panel_scroll_y,
        )

        # Pause butonu (Scroll'dan etkilenmesin diye en üste, panel üzerine çizilir)
        btn_color = (180, 70, 70) if is_paused else (70, 160, 90)
        pygame.draw.rect(screen, btn_color, pause_btn_rect, border_radius=6)
        pygame.draw.rect(screen, (255, 255, 255), pause_btn_rect, 1, border_radius=6)
        btn_text = "► BAŞLAT" if is_paused else "॥ DURDUR"
        btn_surf = font_med.render(btn_text, True, (255, 255, 255))
        bx = pause_btn_rect.x + (pause_btn_rect.width - btn_surf.get_width()) // 2
        by = pause_btn_rect.y + (pause_btn_rect.height - btn_surf.get_height()) // 2
        screen.blit(btn_surf, (bx, by))

        # Ekrana DURAKLATILDI yazısı bas
        if is_paused:
            pause_surface = pygame.Surface((SIM_W, WIN_H), pygame.SRCALPHA)
            pygame.draw.rect(pause_surface, (0, 0, 0, 140), (0, 0, SIM_W, WIN_H))
            font_huge = pygame.font.SysFont("consolas", 48, bold=True)
            text_surf = font_huge.render("DURAKLATILDI", True, (255, 220, 80))
            pause_surface.blit(
                text_surf, 
                ((SIM_W - text_surf.get_width())//2, (WIN_H - text_surf.get_height())//2)
            )
            # Altına bir de ipucu ekleyelim
            hint_surf = font_med.render("(Devam etmek için SPACE veya butona basın)", True, (200, 200, 200))
            pause_surface.blit(
                hint_surf,
                ((SIM_W - hint_surf.get_width())//2, (WIN_H - text_surf.get_height())//2 + 60)
            )
            screen.blit(pause_surface, (0, 0))

        # Scroll göstergesi (panel içeriği ekrandan uzunsa)
        if panel_content_h > WIN_H:
            max_scroll = panel_content_h - WIN_H
            bar_h = max(20, int(WIN_H * WIN_H / panel_content_h))
            bar_y = int(panel_scroll_y / max_scroll * (WIN_H - bar_h))
            pygame.draw.rect(
                screen, (60, 65, 100),
                (WIN_W - 6, bar_y, 4, bar_h), border_radius=2,
            )

        fps_label: str = f"FPS: {int(clock.get_fps())}"
        screen.blit(
            font_sm.render(fps_label, True, (120, 130, 150)), (8, 8)
        )
        pygame.display.flip()

    pygame.quit()


# ── SIMÜLASYONU ÇALIŞTIR ──────────────────────────────────────────────────────
def run_simulation() -> list[HistoryRecord]:
    print("=== GELİŞMİŞ WSN SİMÜLASYONU (Pygame) ===")
    print(f"Sensör Sayısı: {NUM_SENSORS}  |  CH Sayısı: {NUM_CHS}  |  Süre: {SIMULATION_TIME} sn")
    env: simpy.Environment = simpy.Environment()
    sim: WSN_Simulation = WSN_Simulation(env)
    sim.init_network()
    for i in range(NUM_SENSORS):
        env.process(sim.sensor_node(f"Sensor_{i}"))
    for ch_id in sim.ch_ids:
        env.process(sim.cluster_head_process(ch_id))
    env.process(sim.monitor())
    env.run(until=SIMULATION_TIME)

    print("\n--- TRAFİK VE ENERJİ ANALİZİ ---")
    rp: int = sim.total_packets_sent_raw
    ap: int = sim.total_packets_sent_agg
    re: float = sim.total_energy_consumed_raw
    ae: float = sim.total_energy_consumed_agg
    print(f"Ham Paket     : {rp}")
    print(f"Agg Paket     : {ap}")
    if rp > 0:
        print(f"Trafik Azalma : %{(rp - ap) / rp * 100:.2f}")
    print(f"Ham Enerji    : {re:.2f} mJ")
    print(f"Agg Enerji    : {ae:.2f} mJ")
    return sim.history


if __name__ == "__main__":
    history_data: list[HistoryRecord] = run_simulation()
    run_pygame_visualization(history_data)
