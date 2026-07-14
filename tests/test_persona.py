from app import persona


def test_new_merchant_tagged_new(drmeera):
    m = dict(drmeera)
    m["identity"] = dict(drmeera["identity"], established_year=2026)
    assert "new" in persona.tags(m)


def test_established_merchant_tagged_established(drmeera):
    assert "established" in persona.tags(drmeera)  # established_year=2018 in seed data


def test_inactive_persona_from_stale_posts_signal(drmeera):
    # drmeera's seed data carries a "stale_posts:22d" signal
    assert persona.primary(drmeera) == "inactive"


def test_persona_bonus_only_applies_to_mapped_pairs():
    assert persona.persona_bonus("inactive", "reengagement") > 0
    assert persona.persona_bonus("inactive", "compliance") == 0
    assert persona.persona_bonus("busy", "engagement_cadence") < 0


def test_motivator_lookup_is_category_specific():
    assert persona.motivator_for("restaurants") == "footfall"
    assert persona.motivator_for("gyms") == "retention"
    assert persona.motivator_for("totally_unknown_category") == "generic_growth"


def test_none_merchant_returns_empty_tags():
    assert persona.tags(None) == set()
    assert persona.primary(None) == "established"
