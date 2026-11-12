"""Additional cue seeds for fixed source images; one seed role is held out."""
import hashlib
import json

import torch

from .balanced_cues import fit_image_cue


SEED_STRIDE = 10_000_000


def features_for_sources(initial_state, dataset, sources, seeds, feature_indices, lr, device):
    """Preserve source order and reset VAE/Adam independently for every cue."""
    if len(sources) != len(seeds):
        raise ValueError("Source images and seeds must align")
    features = torch.empty(len(sources), len(feature_indices))
    for position, (index, seed) in enumerate(zip(sources.tolist(), seeds.tolist())):
        cue = fit_image_cue(initial_state, dataset[index][0], lr, device, seed)
        features[position] = cue.index_select(0, feature_indices.to(device)).cpu()
        if (position + 1) % 50 == 0 or position + 1 == len(sources):
            print(f"  generated {position + 1}/{len(sources)} cues", flush=True)
    return features


def prepare_seed_variants(base_cache, existing, initial_state, dataset, feature_indices,
                          lr, device, variants, persist):
    """Generate missing per-digit roles, saving after each completed role.

    Offset 0 is reused directly from the original cache. Offsets 1..K-1 train;
    offset K is held out. persist(cache) never changes the base cache.
    """
    if variants < 2:
        raise ValueError("Seed-variant training needs at least two training variants")
    recipe = {"base_signature": base_cache["signature"], "training_variants": variants,
              "seed_stride": SEED_STRIDE, "heldout_offset": variants * SEED_STRIDE}
    signature = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    expanded = existing if existing is not None else {"signature": signature, "recipe": recipe, "digits": []}
    if expanded["signature"] != signature:
        raise ValueError("Seed cache has different base cues or variant settings; choose another prefix")
    for position, base in enumerate(base_cache["digits"]):
        original = base["train"]
        if position == len(expanded["digits"]):
            expanded["digits"].append({"digit": base["digit"], "sources": original["sources"].clone(),
                                        "original_seeds": original["seeds"].clone(), "roles": {}})
        row = expanded["digits"][position]
        if (row["digit"] != base["digit"] or not torch.equal(row["sources"], original["sources"])
                or not torch.equal(row["original_seeds"], original["seeds"])):
            raise ValueError("Seed cache source images or original seeds do not match the base cache")
        for role in range(1, variants + 1):
            name = "heldout_seed" if role == variants else f"train_seed_{role + 1}"
            if name in row["roles"]:
                continue
            seeds = original["seeds"] + role * SEED_STRIDE
            print(f"Digit {base['digit']}, {name}: {len(seeds)} identical source images", flush=True)
            row["roles"][name] = {
                "seeds": seeds,
                "features": features_for_sources(initial_state, dataset, original["sources"], seeds,
                                                  feature_indices, lr, device)}
            persist(expanded)
    return expanded


def assemble_cue_banks(base_cache, expanded=None):
    """Assemble aligned feature/source/seed/target rows; held-out seeds stay separate."""
    variants = expanded["recipe"]["training_variants"] if expanded is not None else 1
    chunks = {"train": [], "validation": []}
    if expanded is not None:
        chunks["heldout_seed"] = []
    for position, base in enumerate(base_cache["digits"]):
        digit = base["digit"]
        original = base["train"]
        chunks["train"].append((digit, original))
        chunks["validation"].append((digit, base["validation"]))
        if expanded is not None:
            row = expanded["digits"][position]
            for role in range(1, variants):
                block = row["roles"][f"train_seed_{role + 1}"]
                chunks["train"].append((digit, {**block, "sources": original["sources"]}))
            heldout = row["roles"]["heldout_seed"]
            chunks["heldout_seed"].append((digit, {**heldout, "sources": original["sources"]}))
    banks = {}
    for split, rows in chunks.items():
        banks[split] = {key: torch.cat([block[key] for _, block in rows])
                        for key in ("features", "sources", "seeds")}
        banks[split]["targets"] = torch.cat([
            torch.full((len(block["features"]),), digit, dtype=torch.long) for digit, block in rows])
    return banks
