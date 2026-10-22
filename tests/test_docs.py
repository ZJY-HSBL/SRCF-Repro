from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_github_math_is_real_latex_not_svg_or_raw_code():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    section=text.split('## 核心公式 · Method',1)[1].split('## 数据与实测',1)[0]
    assert section.count('$$') >= 8 and section.count('$$') % 2 == 0
    assert '\\operatorname{Sim}(u,v)' in section
    assert '\\widehat r_{ui}' in section
    assert '\\frac{' in section
    assert '```math' not in section
    assert '![SR-CF fusion' not in section
    assert 'fusion_equation.svg' not in section


def test_release_after_ci_and_confidence_workflow():
    wf=(ROOT/'.github/workflows/ci-release.yml').read_text(encoding='utf-8')
    research=(ROOT/'.github/workflows/confidence-study.yml').read_text(encoding='utf-8')
    assert "startsWith(github.event.head_commit.message, 'release: v')" in wf
    assert 'experiment: v0.5.0' in research
    assert 'python -m pytest' in research
    assert 'release: v0.5.0' in research

def test_math_section_contains_only_algorithm_content_not_renderer_commentary():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    section=text.split('## 核心公式 · Method',1)[1].split('## 数据与实测',1)[0]
    assert 'GitHub 官方支持的 LaTeX' not in section
    assert 'GitHub Web renders' not in section
    assert '公式不使用截图' not in section
    assert 'mobile apps can differ' not in section
