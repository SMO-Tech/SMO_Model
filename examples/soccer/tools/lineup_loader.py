"""
Utilities for loading lineup CSVs produced by the lineup OCR tool.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RosterEntry:
    jersey_number: Optional[str]
    player_name: str
    position: Optional[str]
    team_id: int
    team_name: str

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "jersey_number": self.jersey_number,
            "player_name": self.player_name,
            "position": self.position,
            "team_id": self.team_id,
            "team_name": self.team_name,
        }


def load_lineup_csv(path: Path, team_name: str, team_id: int) -> List[RosterEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Lineup file not found: {path}")

    entries: List[RosterEntry] = []
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            name = (row.get("name") or row.get("player_name") or "").strip()
            if not name:
                continue
            jersey_number = (row.get("jersey_number") or row.get("number") or "").strip()
            jersey_number = jersey_number if jersey_number else None
            position = (row.get("position") or row.get("player_position") or "").strip()
            position = position if position else None
            entries.append(
                RosterEntry(
                    jersey_number=jersey_number,
                    player_name=name,
                    position=position,
                    team_id=team_id,
                    team_name=team_name,
                )
            )
    return entries


def roster_to_lookup(entries: List[RosterEntry]) -> Dict[str, Dict[str, Optional[str]]]:
    lookup: Dict[str, Dict[str, Optional[str]]] = {}
    for entry in entries:
        if entry.jersey_number:
            lookup[entry.jersey_number] = entry.to_dict()
    return lookup

