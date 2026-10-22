from pathlib import Path


def test_github_markdown_math_has_svg_fallbacks():
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text(encoding="utf-8")
    assert text.count('```math\n') == 2
    assert '\\operatorname{sim}(u,v)' in text
    assert '\\widehat{r}_{ui}' in text
    for f in ('fusion_equation.svg', 'normalized_prediction.svg'):
        svg = root / 'artifacts' / 'figures' / f
        assert f in text
        assert svg.is_file()
        assert '<svg' in svg.read_text(encoding='utf-8')


def test_study_publishing_is_tied_to_verified_ci():
    root = Path(__file__).resolve().parents[1]
    text = (root / '.github/workflows/repeated-research.yml').read_text()
    assert 'git commit -m \'release: v0.3.0' in text
    assert "python -m pytest" in text
    assert '--data-dir /tmp/srcf-original/ml-100k' in text
    assert 'startsWith(github.event.head_commit.message' in text
