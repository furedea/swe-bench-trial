# swe-bench-trial

SWE-bench Lite で Claude Sonnet 4.6 の推論性能を試すトライアルプロジェクト．

## 概要

- **Phase 1 (Inference)**: ローカルで推論を実行しパッチを生成（agent / one_shot の2モード）
- **Phase 2 (Evaluation)**: GitHub Actions (ubuntu-latest / x86_64) で swebench ハーネスを実行

## 構成

```
src/
  dataset.py    # SWE-bench Lite からインスタンス取得・リポジトリ clone
  retrieval.py  # BM25 による関連ファイル検索（one_shot モード用）
  prompt.py     # プロンプト構築（one_shot モード用）
  one_shot.py   # ファイル検索 → プロンプト構築 → LLM 推論パイプライン
  agent.py      # mini-swe-agent を起動し git diff を回収
  model.py      # モデル名のプロバイダプレフィックス正規化
  main.py       # CLI エントリーポイント → outputs/ に保存
outputs/
  predictions_agent.jsonl    # agent モードの生成済みパッチ
  predictions_one_shot.jsonl # one_shot モードの生成済みパッチ
.github/workflows/
  evaluate.yml               # swebench 評価ワークフロー
```

## セットアップ

```bash
uv sync
```

## Phase 1: パッチ生成

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

**agent モード**（mini-swe-agent を使用）:

```bash
uv run --env-file .env python src/main.py --mode agent
```

**one_shot モード**（BM25 ファイル検索 + LLM 一発推論）:

```bash
uv run --env-file .env python src/main.py --mode one_shot --top-k 10
```

完了すると `outputs/predictions_{mode}.jsonl` に以下の形式で保存される:

```json
{"instance_id": "astropy__astropy-12907", "model_name_or_path": "claude-sonnet-4-6", "model_patch": "diff --git ..."}
```

その他のオプション:

```
--instance-id  対象インスタンス ID（デフォルト: astropy__astropy-12907）
--model        使用モデル（デフォルト: claude-sonnet-4-6）
--workspace    リポジトリ clone 先ディレクトリ（デフォルト: workspace/）
--output       出力パス（デフォルト: outputs/predictions_{mode}.jsonl）
```

## Phase 2: 評価 (GitHub Actions)

`outputs/predictions_agent.jsonl` または `outputs/predictions_one_shot.jsonl` を push するか，
手動でモードを指定してワークフローを実行する:

```bash
gh workflow run evaluate.yml --repo furedea/swe-bench-trial -f mode=agent
gh workflow run evaluate.yml --repo furedea/swe-bench-trial -f mode=one_shot
```

結果をローカルで確認:

```bash
gh run list --workflow evaluate.yml --repo furedea/swe-bench-trial
gh run view --job=<job-id> --log --repo furedea/swe-bench-trial | grep -A 30 "Print results"

# またはアーティファクトをダウンロード
gh run download <run-id> --repo furedea/swe-bench-trial --name eval-results-agent --dir eval-results
```

## 評価結果

| instance_id | model | mode | resolved |
|---|---|---|---|
| astropy__astropy-12907 | claude-sonnet-4-6 | agent | true |
| astropy__astropy-12907 | claude-sonnet-4-6 | one_shot | - |

`resolved: true` の基準: `FAIL_TO_PASS` が全て成功 かつ `PASS_TO_FAIL` がゼロ．

## テスト

```bash
uv run --frozen pytest
```

## 注意

- **Inference はローカル実行**（mini-swe-agent, Docker 不要）
- **Evaluation は Actions 実行**（x86_64 Linux + Docker が必要なため Apple Silicon では非推奨）
- `workspace/` と `.env` は gitignore 済み
