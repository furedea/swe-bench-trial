# swe-bench-trial

SWE-bench Lite データセットから1つのインスタンスを選び，Claude Sonnet 4.6 でパッチ生成から評価までの一連の流れを試すトライアルプロジェクト．
デフォルトでは `astropy__astropy-12907`（astropy の separability matrix バグ）を対象とする．

## 概要

- **Phase 1 (Inference)**: ローカルで推論を実行しパッチを生成（agent / one_shot の2モード）
- **Phase 2 (Evaluation)**: GitHub Actions (ubuntu-latest / x86_64) で swebench ハーネスを実行

## 構成

```
src/
  main.py       # CLI エントリーポイント（argparse → パッチ生成 → JSONL 保存）
  dataset.py    # SWEInstance / SWETask 定義，インスタンス取得，リポジトリ clone
  retrieval.py  # BM25 による関連ファイル検索（one_shot モード用）
  prompt.py     # プロンプト構築（テンプレート読み込み + 行番号付きファイル表示）
  one_shot.py   # ファイル検索 → プロンプト構築 → LLM 推論 → diff 抽出
  agent.py      # mini-swe-agent 実行（コマンド）+ git diff 回収（クエリ）
  model.py      # モデル名のプロバイダプレフィックス正規化
prompts/
  prompt_template.txt        # one_shot 用プロンプトテンプレート（prompt_style_2 準拠）
outputs/
  predictions_agent.jsonl    # agent モードの生成済みパッチ
  predictions_one_shot.jsonl # one_shot モードの生成済みパッチ
eval-results-one-shot/       # one_shot 評価結果（swebench report.json 等）
.github/workflows/
  evaluate.yml               # swebench 評価ワークフロー（agent / one_shot matrix）
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
uv run --env-file .env python src/main.py --mode one_shot
```

完了すると `outputs/predictions_{mode}.jsonl` に以下の形式で保存される:

```json
{"instance_id": "astropy__astropy-12907", "model_name_or_path": "claude-sonnet-4-6", "model_patch": "diff --git ..."}
```

### パッチ生成の工程

#### 共通工程（agent / one_shot 共通）

```
main.py
  1. インスタンス取得   dataset.load_instance()
  2. リポジトリ準備     dataset.setup_repo()
  3. パッチ生成         one_shot.run_one_shot() or agent.run_agent()
  4. 保存               main.save_prediction()
```

1. **インスタンス取得** (`dataset.load_instance`): HuggingFace の `princeton-nlp/SWE-bench_Lite` (test split) から対象インスタンスのメタデータ (`instance_id`, `repo`, `base_commit`, `problem_statement`) を取得
2. **リポジトリ準備** (`dataset.setup_repo`): GitHub からリポジトリを clone し，`git reset --hard {base_commit}` + `git clean -fd` でベースコミット時点の状態に復元
3. **パッチ生成**: モード別（後述）
4. **保存** (`main.save_prediction`): SWE-bench 形式の JSONL レコードとして `outputs/predictions_{mode}.jsonl` に追記

#### one_shot モード

```
one_shot.run_one_shot()
  ├─ retrieval.retrieve_files()   BM25 で関連ファイル検索
  ├─ prompt.build_prompt()        行番号付きコードをテンプレートに埋め込み
  ├─ litellm.completion()         Claude API 呼び出し
  └─ one_shot.extract_diff()      レスポンスから diff 抽出
```

1. **ファイル検索** (`retrieval.retrieve_files`): リポジトリ内の全 `.py` ファイルを対象に BM25Okapi で `problem_statement` との関連度をスコアリングし，上位 `top_k` 件を取得
2. **プロンプト構築** (`prompt.build_prompt`): 取得したファイルに行番号を付与し，`prompts/prompt_template.txt` のテンプレートに埋め込む．テンプレートは Jimenez et al. 2023 の `prompt_style_2` に準拠し，unified diff 形式の few-shot 例を含む
3. **LLM 推論** (`one_shot._call_model`): litellm 経由で Claude API を呼び出し（model 名は `anthropic/` プレフィックスを自動付与）
4. **diff 抽出** (`one_shot.extract_diff`): モデル出力から以下の優先順位で unified diff を抽出:
   1. `<patch>...</patch>` XML タグ
   2. `` ```diff ... ``` `` コードフェンス
   3. `--- a/...` で始まる生の diff パターン

#### agent モード

```
agent.run_agent()    → mini-swe-agent をサブプロセス実行
agent.collect_patch() → git diff --no-ext-diff HEAD で差分回収
```

1. **agent 実行** (`agent.run_agent`): `mini-swe-agent` をサブプロセスで実行（`--yolo` で確認スキップ，`--exit-immediately` で完了後に即終了）
2. **差分回収** (`agent.collect_patch`): agent がファイル編集を行った後，`git diff --no-ext-diff HEAD` で unified diff を取得

その他のオプション:

```
--instance-id       対象インスタンス ID（デフォルト: astropy__astropy-12907）
--model             使用モデル（デフォルト: claude-sonnet-4-6）
--top-k             one_shot モードで検索するファイル数（デフォルト: 10）
--workspace         リポジトリ clone 先ディレクトリ（デフォルト: workspace/）
--output            出力パス（デフォルト: outputs/predictions_{mode}.jsonl）
--prompt-template   プロンプトテンプレートファイル（デフォルト: prompts/prompt_template.txt）
```

## Phase 2: 評価 (GitHub Actions)

`outputs/predictions_agent.jsonl` または `outputs/predictions_one_shot.jsonl` を push するか，
手動でモードを指定してワークフローを実行する:

```bash
gh workflow run evaluate.yml --repo furedea/swe-bench-trial
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
| astropy__astropy-12907 | claude-sonnet-4-6 | one_shot | true |

`resolved: true` の基準: `FAIL_TO_PASS` が全て成功 かつ `PASS_TO_FAIL` がゼロ．

### 生成パッチの違い（astropy__astropy-12907）

両モードとも `resolved: true` だが，生成されたパッチの内容は異なる:

- **共通**: `astropy/modeling/separable.py` L245 の `= 1` → `= right` への1行修正（バグの核心）
- **one_shot**: 上記の1行修正のみ
- **agent**: 上記に加え，`astropy/modeling/tests/test_separable.py` にテストケース `cm8`（`rot & (sh1 & sh2)` の compound model）を追加

結果が同一になった理由: SWE-bench はリポジトリに元々存在するテスト（FAIL_TO_PASS / PASS_TO_FAIL）の成否で評価するため，agent が追加したテストは評価に影響しない．バグ修正の核心が同一のため両方 `resolved: true` となった．

## テスト

```bash
uv run --frozen pytest
```

## 注意

- **Inference はローカル実行**（mini-swe-agent, Docker 不要）
- **Evaluation は Actions 実行**（x86_64 Linux + Docker が必要なため Apple Silicon では非推奨）
- `workspace/` と `.env` は gitignore 済み
