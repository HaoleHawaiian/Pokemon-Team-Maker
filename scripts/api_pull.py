"""
Long-running fetch of Pokédex entries and types from PokeAPI via pokebase.

Run from repo root: python scripts/api_pull.py --dex-out Data/pokedex_entries.csv

Requires: pip install pokebase pandas
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def fetch_dex_rows() -> list[dict]:
    import pokebase as pb

    dex: list[dict] = []
    pokemon_species_list = pb.APIResourceList("pokemon-species")
    total = pokemon_species_list.count
    start_time = time.time()

    for i in range(total):
        pokemon = pb.pokemon_species(i + 1)
        number = pokemon.id
        name = pokemon.name.capitalize()
        color = pokemon.color.name
        habitat = pokemon.habitat.name if pokemon.habitat else None
        generation = pokemon.generation.name
        egg_group = pokemon.egg_groups[0].name if pokemon.egg_groups else "unknown"

        if (i + 1) % 10 == 0:
            print(name, time.time() - start_time)

        for j in pokemon.flavor_text_entries:
            if j.language.name == "en":
                flavor_text = j.flavor_text.strip().replace("\n", " ").lower()
                dex.append(
                    {
                        "Pokemon": name,
                        "Number": number,
                        "Color": color,
                        "Habitat": habitat,
                        "Generation": generation,
                        "Egg Group": egg_group,
                        "Description": flavor_text,
                    }
                )

    return dex


def fetch_types_rows() -> list[dict]:
    import pokebase as pb

    pokemon_list = pb.APIResourceList("pokemon")
    total = pokemon_list.count
    types: list[dict] = []
    type_time_start = time.time()

    for i in range(total):
        try:
            pokemon_type = pb.pokemon(i + 1)
        except Exception as e:
            print(f"Stopping at id {i + 1}: {e}", file=sys.stderr)
            break
        name = pokemon_type.name.capitalize()
        if (i + 1) % 10 == 0:
            print(name)
        for j in pokemon_type.types:
            tname = j.type.name.strip().capitalize()
            types.append({"Pokemon": name, "Type": tname})
        time.sleep(0.5)

    print("Type fetch seconds:", time.time() - type_time_start)
    return types


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dex-out", type=Path, default=REPO_ROOT / "Data" / "pokedex_entries.csv")
    parser.add_argument("--types-out", type=Path, default=REPO_ROOT / "Data" / "pokemon_types.csv")
    parser.add_argument("--types-only", action="store_true", help="Only fetch pokemon types CSV")
    parser.add_argument("--dex-only", action="store_true", help="Only fetch species / dex CSV")
    args = parser.parse_args()

    if not args.types_only:
        dex = fetch_dex_rows()
        df = pd.DataFrame(dex).drop_duplicates(subset="Description").reset_index(drop=True)
        args.dex_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.dex_out, index=False)
        print(f"Wrote {args.dex_out} rows={len(df)}")

    if not args.dex_only:
        types = fetch_types_rows()
        tdf = pd.DataFrame(types)
        args.types_out.parent.mkdir(parents=True, exist_ok=True)
        tdf.to_csv(args.types_out, index=False)
        print(f"Wrote {args.types_out} rows={len(tdf)}")


if __name__ == "__main__":
    main()
