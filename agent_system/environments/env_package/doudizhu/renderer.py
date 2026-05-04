from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CARD_LABELS = {
    "T": "10",
    "B": "BJ",
    "R": "RJ",
}


@dataclass(frozen=True)
class HitBox:
    kind: str
    box: Tuple[int, int, int, int]
    payload: Optional[int] = None

    def contains(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self.box
        return x0 <= x <= x1 and y0 <= y <= y1


class DoudizhuRenderer:
    """Small in-memory renderer for VL training observations."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = int(width)
        self.height = int(height)
        self.font = self._load_font(12)
        self.big_font = self._load_font(14)

    def norm_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        px = int(round(float(x) / 1000.0 * (self.width - 1)))
        py = int(round(float(y) / 1000.0 * (self.height - 1)))
        return px, py

    def hit_test(self, state: Dict, selected_indices: Sequence[int], x: float, y: float) -> Optional[HitBox]:
        px, py = self.norm_to_pixel(x, y)
        hitboxes = self.get_hitboxes(state, selected_indices)
        for hitbox in reversed(hitboxes):
            if hitbox.contains(px, py):
                return hitbox
        return None

    def get_hitboxes(self, state: Dict, selected_indices: Sequence[int] = ()) -> List[HitBox]:
        hand = state.get("current_hand", "")
        hitboxes: List[HitBox] = []
        card_w, card_h, gap, start_x, base_y = self._hand_layout(hand)
        selected = set(selected_indices)
        for idx, _card in enumerate(hand):
            x0 = start_x + idx * (card_w + gap)
            y0 = base_y - 14 if idx in selected else base_y
            hitboxes.append(HitBox("card", (x0, y0, x0 + card_w, y0 + card_h), idx))

        play_box, pass_box = self._button_boxes(base_y)
        hitboxes.append(HitBox("play", play_box))
        hitboxes.append(HitBox("pass", pass_box))
        return hitboxes

    def render(self, state: Dict, selected_indices: Sequence[int] = (), message: str = "") -> np.ndarray:
        image = Image.new("RGB", (self.width, self.height), (35, 112, 73))
        draw = ImageDraw.Draw(image)

        draw.rectangle((0, 0, self.width, 72), fill=(25, 78, 96))
        role = "Landlord" if state.get("self") == state.get("landlord") else "Peasant"
        counts = state.get("num_cards_left", [0, 0, 0])
        seen = self._pretty_cards(state.get("seen_cards", ""))
        draw.text((14, 12), f"You: Player {state.get('self', 0)} ({role})", fill=(245, 245, 245), font=self.big_font)
        draw.text((self.width - 214, 12), f"Bottom cards: {seen or '-'}", fill=(245, 245, 245), font=self.font)

        p1_box = self._draw_opponent(draw, "P1", self.width - 150, 92, counts[1])
        p2_box = self._draw_opponent(draw, "P2", 24, 92, counts[2])

        p0_anchor = (self.width // 2, self.height - 112)
        p1_anchor = (p1_box[0] + 53, p1_box[3] + 22)
        p2_anchor = (p2_box[0] + 53, p2_box[3] + 22)
        self._draw_turn_arrows(draw, p0_anchor, p1_anchor, p2_anchor)
        self._draw_current_trick(draw, state)

        for hitbox in self.get_hitboxes(state, selected_indices):
            if hitbox.kind == "card":
                self._draw_card(draw, hitbox, state["current_hand"][hitbox.payload], hitbox.payload in set(selected_indices))
            elif hitbox.kind == "play":
                self._draw_button(draw, hitbox.box, "PLAY", (220, 78, 56))
            elif hitbox.kind == "pass":
                self._draw_button(draw, hitbox.box, "PASS", (82, 95, 108))

        draw.text((14, self.height - 25), "Click cards, then PLAY above your hand. Click PASS to skip when allowed.", fill=(236, 242, 230), font=self.font)
        return np.array(image, dtype=np.uint8)

    def _draw_opponent(self, draw: ImageDraw.ImageDraw, label: str, x: int, y: int, count: int):
        box = (x, y, x + 106, y + 76)
        draw.rounded_rectangle(box, radius=6, fill=(229, 233, 218), outline=(28, 75, 58), width=2)
        draw.text((x + 12, y + 10), label, fill=(28, 45, 38), font=self.big_font)
        draw.text((x + 12, y + 34), f"{count} cards", fill=(28, 45, 38), font=self.font)
        return box

    def _draw_card(self, draw: ImageDraw.ImageDraw, hitbox: HitBox, card: str, selected: bool):
        x0, y0, x1, y1 = hitbox.box
        self._draw_card_face(draw, x0, y0, x1 - x0, y1 - y0, card, selected=selected)

    def _draw_card_face(self, draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, card: str, selected: bool = False):
        fill = (255, 248, 214) if not selected else (255, 221, 112)
        outline = (31, 52, 47) if not selected else (208, 67, 48)
        draw.rounded_rectangle((x, y, x + width, y + height), radius=4, fill=fill, outline=outline, width=2)
        label = CARD_LABELS.get(card, card)
        color = (177, 34, 41) if card in ("R", "B", "2") else (22, 31, 34)
        draw.text((x + 6, y + 8), label, fill=color, font=self.big_font)

    def _draw_button(self, draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], label: str, fill: Tuple[int, int, int]):
        draw.rounded_rectangle(box, radius=5, fill=fill, outline=(255, 255, 255), width=1)
        draw.text((box[0] + 18, box[1] + 10), label, fill=(255, 255, 255), font=self.big_font)

    def _load_font(self, size: int):
        for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _hand_layout(self, hand: str):
        card_w = max(24, min(34, (self.width - 28) // max(len(hand), 1) - 3))
        card_h = 58
        gap = 3
        total_w = len(hand) * card_w + max(len(hand) - 1, 0) * gap
        start_x = max(10, (self.width - total_w) // 2)
        base_y = self.height - 92
        return card_w, card_h, gap, start_x, base_y

    def _button_boxes(self, base_y: int):
        button_w = 80
        button_h = 34
        gap = 18
        top = base_y - 42
        center_x = self.width // 2
        play_box = (center_x - gap // 2 - button_w, top, center_x - gap // 2, top + button_h)
        pass_box = (center_x + gap // 2, top, center_x + gap // 2 + button_w, top + button_h)
        return play_box, pass_box

    def _draw_current_trick(self, draw: ImageDraw.ImageDraw, state: Dict):
        plays = self._current_trick_actions(state.get("trace", []))
        self._draw_play_area(draw, "P2", plays.get(2), 36, 218, align="left")
        self._draw_play_area(draw, "P1", plays.get(1), self.width - 214, 218, align="left")
        self._draw_play_area(draw, "P0", plays.get(0), self.width // 2 - 92, 264, align="center")

    def _draw_play_area(self, draw: ImageDraw.ImageDraw, label: str, action: Optional[str], x: int, y: int, align: str):
        area_w = 184
        area_h = 64
        draw.rounded_rectangle((x, y, x + area_w, y + area_h), radius=6, fill=(41, 92, 67), outline=(207, 224, 184), width=1)
        draw.text((x + 8, y + 6), label, fill=(241, 246, 232), font=self.font)
        if action is None:
            draw.text((x + 42, y + 30), "-", fill=(214, 224, 207), font=self.big_font)
        elif action == "pass":
            draw.rounded_rectangle((x + 52, y + 28, x + 132, y + 52), radius=5, fill=(93, 105, 116), outline=(241, 246, 232), width=1)
            draw.text((x + 76, y + 35), "PASS", fill=(255, 255, 255), font=self.font)
        else:
            self._draw_card_row(draw, action, x + 32, y + 24, max_width=140)

    def _draw_card_row(self, draw: ImageDraw.ImageDraw, cards: str, x: int, y: int, max_width: int):
        card_w = 22
        card_h = 32
        gap = 2
        if len(cards) > 0:
            total = len(cards) * card_w + (len(cards) - 1) * gap
            if total > max_width:
                gap = -max(0, (total - max_width) // max(len(cards) - 1, 1))
        for idx, card in enumerate(cards):
            self._draw_card_face(draw, x + idx * (card_w + gap), y, card_w, card_h, card)

    def _current_trick_actions(self, trace: Sequence[Tuple[int, str]]):
        if not trace:
            return {}
        trick_start = 0
        for idx, (_player_id, action) in enumerate(trace):
            if idx == 0:
                trick_start = 0
                continue
            if action != "pass" and idx >= 2 and trace[idx - 1][1] == "pass" and trace[idx - 2][1] == "pass":
                trick_start = idx
        return {player_id: action for player_id, action in trace[trick_start:]}

    def _draw_turn_arrows(self, draw: ImageDraw.ImageDraw, p0: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int]):
        color = (255, 235, 164)
        self._draw_arrow(draw, (p0[0] + 108, p0[1] - 2), (p1[0] - 48, p1[1] + 6), color)
        self._draw_arrow(draw, (p1[0] - 84, p1[1] - 48), (p2[0] + 84, p2[1] - 48), color)
        self._draw_arrow(draw, (p2[0] + 48, p2[1] + 6), (p0[0] - 108, p0[1] - 2), color)

    def _draw_arrow(self, draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int]):
        draw.line((start, end), fill=color, width=3)
        sx, sy = start
        ex, ey = end
        vec = np.array([ex - sx, ey - sy], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            return
        unit = vec / norm
        perp = np.array([-unit[1], unit[0]])
        tip = np.array([ex, ey], dtype=np.float32)
        left = tip - unit * 12 + perp * 6
        right = tip - unit * 12 - perp * 6
        draw.polygon([tuple(tip), tuple(left), tuple(right)], fill=color)

    def _pretty_cards(self, cards: str) -> str:
        if cards == "pass":
            return "PASS"
        return " ".join(CARD_LABELS.get(card, card) for card in cards)
