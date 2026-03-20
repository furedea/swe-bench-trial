# swe-bench-trial

SWE-bench Lite で Claude Sonnet 4.6 の推論性能を試すトライアルプロジェクト．

## 概要

- **Phase 1 (Inference)**: ローカルで mini-swe-agent を使いパッチを生成
- **Phase 2 (Evaluation)**: GitHub Actions (ubuntu-latest / x86_64) で swebench ハーネスを実行

## 構成

```
src/
  dataset.py   # SWE-bench Lite からインスタンス取得・リポジトリ clone
  runner.py    # mini-swe-agent を起動し git diff を回収
  main.py      # オーケストレーター → outputs/predictions.jsonl に保存
outputs/
  predictions.jsonl  # 生成済みパッチ（SWE-bench 準拠フォーマット）
.github/workflows/
  evaluate.yml       # swebench 評価ワークフロー
```

## セットアップ

```bash
uv sync
```

## Phase 1: パッチ生成

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
uv run --env-file .env python src/main.py
```

完了すると `outputs/predictions.jsonl` に以下の形式で保存される:

```json
{"instance_id": "astropy__astropy-12907", "model_name_or_path": "claude-sonnet-4-6", "model_patch": "diff --git ..."}
```

## Phase 2: 評価 (GitHub Actions)

`outputs/predictions.jsonl` を push するか，手動でワークフローを実行する:

```bash
gh workflow run evaluate.yml --repo furedea/swe-bench-trial
```

結果をローカルで確認:

```bash
# ログから直接読む（ダウンロード不要）
gh run list --workflow evaluate.yml --repo furedea/swe-bench-trial
gh run view --job=<job-id> --log --repo furedea/swe-bench-trial | grep -A 30 "Print results"

# またはアーティファクトをダウンロード
gh run download <run-id> --repo furedea/swe-bench-trial --name eval-results --dir eval-results
```

## 評価結果

| instance_id | model | resolved |
|---|---|---|
| astropy__astropy-12907 | claude-sonnet-4-6 | ✅ true |

`resolved: true` の基準: `FAIL_TO_PASS` が全て成功 かつ `PASS_TO_FAIL` がゼロ．

## テスト

```bash
uv run --frozen pytest
```

## 注意

- **Inference はローカル実行**（mini-swe-agent, Docker 不要）
- **Evaluation は Actions 実行**（x86_64 Linux + Docker が必要なため Apple Silicon では非推奨）
- `workspace/` と `.env` は gitignore 済み
