from pathlib import Path


def test_github_readme_formula_images_do_not_expose_latex_source():
    root = Path(__file__).resolve().parents[1]
    text = (root / 'README.md').read_text(encoding='utf-8')
    # GitHub iOS displays ```math as raw code. Keep formula images, not fences.
    assert '```math' not in text
    assert '$$' not in text
    assert '\\widehat{' not in text
    for f in ('fusion_equation.svg', 'normalized_prediction.svg'):
        svg = root / 'artifacts' / 'figures' / f
        assert f in text
        assert svg.is_file()
        xml = svg.read_text(encoding='utf-8')
        assert '<svg' in xml
        assert 'style="fill: #ffffff"' in xml


def test_release_and_followup_are_behind_ci():
    root = Path(__file__).resolve().parents[1]
    main = (root / '.github/workflows/ci-release.yml').read_text()
    followup = (root / '.github/workflows/seed-followup.yml').read_text()
    assert "startsWith(github.event.head_commit.message, 'release: v')" in main
    assert 'experiment: v0.4.0' in followup
    assert 'python -m pytest' in followup
    assert 'seed_diagnostics' in followup
    assert 'release: v0.4.0' in followup
