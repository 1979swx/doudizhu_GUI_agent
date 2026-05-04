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
        self.font = ImageFont.load_default()
        self.big_font = ImageFont.load_default()

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
        card_w = max(24, min(34, (self.width - 28) // max(len(hand), 1) - 3))
        card_h = 58
        gap = 3
        total_w = len(hand) * card_w + max(len(hand) - 1, 0) * gap
        start_x = max(10, (self.width - total_w) // 2)
        base_y = self.height - 92
        selected = set(selected_indices)
        for idx, _card in enumerate(hand):
            x0 = start_x + idx * (card_w + gap)
            y0 = base_y - 14 if idx in selected else base_y
            hitboxes.append(HitBox("card", (x0, y0, x0 + card_w, y0 + card_h), idx))

        button_y = self.height - 34
        hitboxes.append(HitBox("play", (self.width - 172, button_y - 34, self.width - 92, button_y)))
        hitboxes.append(HitBox("pass", (self.width - 84, button_y - 34, self.width - 8, button_y)))
        return hitboxes

    def render(self, state: Dict, selected_indices: Sequence[int] = (), message: str = "") -> np.ndarray:
        image = Image.new("RGB", (self.width, self.height), (35, 112, 73))
        draw = ImageDraw.Draw(image)

        draw.rectangle((0, 0, self.width, 72), fill=(25, 78, 96))
        role = "Landlord" if state.get("self") == state.get("landlord") else "Peasant"
        counts = state.get("num_cards_left", [0, 0, 0])
        seen = self._pretty_cards(state.get("seen_cards", ""))
        draw.text((14, 12), f"You: Player {state.get('self', 0)} ({role})", fill=(245, 245, 245), font=self.big_font)
        draw.text((14, 36), f"Bottom cards: {seen or '-'}", fill=(228, 238, 238), font=self.font)
        draw.text((self.width - 214, 12), f"Cards left  P0:{counts[0]}  P1:{counts[1]}  P2:{counts[2]}", fill=(245, 245, 245), font=self.font)

        self._draw_opponent(draw, "P1", self.width - 150, 104, counts[1])
        self._draw_opponent(draw, "P2", 24, 104, counts[2])

        trace = state.get("trace", [])
        last_lines = trace[-5:]
        draw.rounded_rectangle((150, 104, self.width - 150, 218), radius=6, fill=(232, 239, 217), outline=(20, 72, 52), width=2)
        draw.text((164, 116), "Recent plays", fill=(20, 40, 34), font=self.font)
        y = 138
        if last_lines:
            for player_id, action in last_lines:
                draw.text((164, y), f"P{player_id}: {self._pretty_cards(action)}", fill=(24, 48, 40), font=self.font)
                y += 16
        else:
            draw.text((164, y), "No cards have been played yet.", fill=(24, 48, 40), font=self.font)

        if message:
            draw.text((14, 78), message[:90], fill=(255, 236, 175), font=self.font)

        for hitbox in self.get_hitboxes(state, selected_indices):
            if hitbox.kind == "card":
                self._draw_card(draw, hitbox, state["current_hand"][hitbox.payload], hitbox.payload in set(selected_indices))
            elif hitbox.kind == "play":
                self._draw_button(draw, hitbox.box, "PLAY", (220, 78, 56))
            elif hitbox.kind == "pass":
                self._draw_button(draw, hitbox.box, "PASS", (82, 95, 108))

        draw.text((14, self.height - 25), "Click cards, then PLAY. Click PASS to skip when allowed.", fill=(236, 242, 230), font=self.font)
        return np.array(image, dtype=np.uint8)

    def _draw_opponent(self, draw: ImageDraw.ImageDraw, label: str, x: int, y: int, count: int):
        draw.rounded_rectangle((x, y, x + 106, y + 76), radius=6, fill=(229, 233, 218), outline=(28, 75, 58), width=2)
        draw.text((x + 12, y + 10), label, fill=(28, 45, 38), font=self.big_font)
        draw.text((x + 12, y + 34), f"{count} cards", fill=(28, 45, 38), font=self.font)

    def _draw_card(self, draw: ImageDraw.ImageDraw, hitbox: HitBox, card: str, selected: bool):
        x0, y0, x1, y1 = hitbox.box
        fill = (255, 248, 214) if not selected else (255, 221, 112)
        outline = (31, 52, 47) if not selected else (208, 67, 48)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=4, fill=fill, outline=outline, width=2)
        label = CARD_LABELS.get(card, card)
        color = (177, 34, 41) if card in ("R", "B", "2") else (22, 31, 34)
        draw.text((x0 + 6, y0 + 8), label, fill=color, font=self.big_font)

    def _draw_button(self, draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], label: str, fill: Tuple[int, int, int]):
        draw.rounded_rectangle(box, radius=5, fill=fill, outline=(255, 255, 255), width=1)
        draw.text((box[0] + 18, box[1] + 10), label, fill=(255, 255, 255), font=self.big_font)

    def _pretty_cards(self, cards: str) -> str:
        if cards == "pass":
            return "PASS"
        return " ".join(CARD_LABELS.get(card, card) for card in cards)
