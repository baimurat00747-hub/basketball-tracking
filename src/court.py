"""Модель разметки площадки ФИБА 28x15 и отрисовка схемы.

Мировые координаты: начало в левом нижнем углу (вид сверху),
X вдоль длинной стороны (0..28), Y вдоль короткой (0..15), метры.
Используется для гомографии (этап 4) и схем/тепловых карт (этап 6).
"""
from __future__ import annotations

import numpy as np

from src.config import (
    COURT_LENGTH_M as L,
    COURT_WIDTH_M as W,
    FIBA_PAINT_LENGTH_M as PAINT_L,
    FIBA_PAINT_WIDTH_M as PAINT_W,
    FIBA_ARC_RADIUS_M as ARC_R,
    FIBA_ARC_SIDE_OFFSET_M as ARC_OFF,
    FIBA_HOOP_FROM_BASELINE_M as HOOP_X,
    FIBA_CENTER_CIRCLE_R_M as CC_R,
    FIBA_FT_CIRCLE_R_M as FT_R,
)


def _arc(cx, cy, r, a1, a2, n=60):
    t = np.linspace(np.radians(a1), np.radians(a2), n)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def court_polylines() -> list[np.ndarray]:
    """Разметка площадки как список полилиний (N, 2) в метрах."""
    cy = W / 2  # ось колец
    lines: list[np.ndarray] = [
        # внешний прямоугольник
        np.array([(0, 0), (L, 0), (L, W), (0, W), (0, 0)], float),
        # центральная линия
        np.array([(L / 2, 0), (L / 2, W)], float),
        # центральный круг
        _arc(L / 2, cy, CC_R, 0, 360),
    ]
    for side in (0, 1):  # 0 = левая корзина, 1 = правая
        def x(v):  # зеркалирование по стороне
            return v if side == 0 else L - v

        y1, y2 = cy - PAINT_W / 2, cy + PAINT_W / 2
        # краска (прямоугольник)
        lines.append(np.array(
            [(x(0), y1), (x(PAINT_L), y1), (x(PAINT_L), y2), (x(0), y2)], float))
        # полукруг у штрафной (в сторону центра)
        a1, a2 = (-90, 90) if side == 0 else (90, 270)
        lines.append(_arc(x(PAINT_L), cy, FT_R, a1, a2))
        # трёхочковая: дуга + прямые участки до лицевой
        # дуга от точки касания прямых участков (y = ARC_OFF от боковых)
        dy = W - 2 * ARC_OFF  # не используется; для ясности границ
        # угол, при котором дуга достигает y = cy +/- (cy - ARC_OFF)
        reach = cy - ARC_OFF
        ang = np.degrees(np.arcsin(min(reach / ARC_R, 1.0)))
        if side == 0:
            arc = _arc(HOOP_X, cy, ARC_R, -ang, ang)
        else:
            arc = _arc(L - HOOP_X, cy, ARC_R, 180 - ang, 180 + ang)
        # x-координата конца дуги
        x_end = HOOP_X + ARC_R * np.cos(np.radians(ang))
        lines.append(arc)
        lines.append(np.array([(x(0), ARC_OFF), (x(x_end if side == 0 else x_end), ARC_OFF)], float)
                     if side == 0 else
                     np.array([(L - x_end, ARC_OFF), (L, ARC_OFF)], float))
        if side == 0:
            lines.append(np.array([(0, W - ARC_OFF), (x_end, W - ARC_OFF)], float))
        else:
            lines.append(np.array([(L - x_end, W - ARC_OFF), (L, W - ARC_OFF)], float))
    return lines


def draw_court(ax, color="black", lw=1.2):
    """Нарисовать схему площадки на matplotlib-оси (вид сверху, метры)."""
    for poly in court_polylines():
        ax.plot(poly[:, 0], poly[:, 1], color=color, lw=lw)
    ax.set_xlim(-1, L + 1)
    ax.set_ylim(-1, W + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    return ax


def project_polylines(H: np.ndarray, width: int = 1280, height: int = 720) -> list[np.ndarray]:
    """Спроецировать разметку (метры -> пиксели кадра) гомографией H_world2img.

    Возвращает полилинии в пикселях, обрезанные по видимости (точки за кадром
    с запасом 200px отбрасываются, полилиния рвётся на сегменты).
    """
    out: list[np.ndarray] = []
    for poly in court_polylines():
        pts = np.hstack([poly, np.ones((len(poly), 1))])
        proj = (H @ pts.T).T
        z = proj[:, 2]
        # точки за камерой (z<=0) невалидны
        valid = z > 1e-6
        xy = np.full((len(poly), 2), np.nan)
        xy[valid] = proj[valid, :2] / z[valid, None]
        inside = valid & (xy[:, 0] > -200) & (xy[:, 0] < width + 200) \
                       & (xy[:, 1] > -200) & (xy[:, 1] < height + 200)
        # рвём на видимые сегменты
        seg = []
        for p, ok in zip(xy, inside):
            if ok:
                seg.append(p)
            elif len(seg) > 1:
                out.append(np.array(seg)); seg = []
            else:
                seg = []
        if len(seg) > 1:
            out.append(np.array(seg))
    return out
