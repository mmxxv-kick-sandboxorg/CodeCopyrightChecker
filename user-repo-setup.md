# ユーザーリポジトリセットアップ手順

このドキュメントでは、組織のCodeCopyrightCheckerアクションを使用するために、ユーザーリポジトリに必要な設定を行う手順を説明します。

## 前提条件

- 組織側でCodeCopyrightCheckerリポジトリが既にセットアップされていること
- ユーザーが組織のメンバーであること
- リポジトリに対する適切な権限を持っていること

## 1. リポジトリの準備

### 1.1 GitHub Actions の有効化
1. リポジトリの **Settings** タブに移動
2. 左サイドバーの **Actions** > **General** をクリック
3. **Actions permissions** で適切な設定を選択（推奨: "Allow all actions and reusable workflows"）

## 2. ワークフローファイルの作成

### 2.1 ディレクトリ構造の作成
リポジトリのルートディレクトリに以下の構造を作成：

```
your-repo/
├── .github/
│   └── workflows/
│       └── checkinfringement.yml
└── (その他のプロジェクトファイル)
```

### 2.2 checkinfringement.yml の作成
`.github/workflows/checkinfringement.yml` ファイルを作成し、以下の内容を記述：

```yaml
# .github/workflows/checkinfringement.yml
name: Run CodeCopyrightInfringementCheck on PR

# PR 作成・更新時に発火して、composite action を呼び出す
on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write   # PRにコメントするために必要
  issues: write          # issueコメントを使う場合に必要
  packages: write        # GHCR への push に必要

jobs:
  copyright-check:
    runs-on: ubuntu-latest
    steps:
      - name: Run copyright check
        uses: YOUR_ORG_NAME/CodeCopyrightChecker@main  # 組織名を実際の名前に変更
        with:
          pr_number: ${{ github.event.pull_request.number }}
          GH_TOKEN: ${{ secrets.GH_TOKEN }}
          HUGGINGFACE_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
          AACS: ${{ secrets.AACS }}
          AACSAPIkey: ${{ secrets.AACSAPIkey }}
```

**注意**: `YOUR_ORG_NAME` を実際の組織名に変更してください。

## 3. Secrets の設定

### 3.1 Personal Access Token (PAT) の作成
1. GitHub の **Settings** > **Developer settings** > **Personal access tokens** > **Fine-grained tokens** に移動
2. **Generate new token** をクリック
3. 以下の権限を設定：
   - `contents:read`
   - `packages:write`
   - `packages:read`
   - `pull-requests:write`
4. トークンを生成し、安全な場所に保存

### 3.2 リポジトリSecrets の設定
リポジトリの **Settings** > **Secrets and variables** > **Actions** で以下のSecretsを追加：

| Secret名 | 説明 | 取得方法 |
|---------|------|----------|
| `GH_TOKEN` | GitHub Personal Access Token | 上記で作成したPAT |
| `HUGGINGFACE_TOKEN` | HuggingFace APIトークン | [HuggingFace Settings](https://huggingface.co/settings/tokens)で作成 |
| `AACS` | Azure Content Safety エンドポイントURL | Azure portalから取得 |
| `AACSAPIkey` | Azure Content Safety APIキー | Azure portalから取得 |

### 3.3 各Secretの取得方法詳細

#### HuggingFace Token
1. [HuggingFace](https://huggingface.co/) にログイン
2. **Settings** > **Access Tokens** に移動
3. **New token** をクリックして作成

#### Azure Content Safety
1. [Azure Portal](https://portal.azure.com/) にログイン
2. **Content Safety** サービスを作成
3. **Keys and Endpoint** からエンドポイントURLとAPIキーを取得

## 4. テスト実行

### 4.1 プルリクエストの作成
1. 新しいブランチを作成
2. 何らかのファイルを変更
3. プルリクエストを作成
4. GitHub Actions が自動的に実行されることを確認

### 4.2 動作確認
- **Actions** タブでワークフローの実行状況を確認
- エラーが発生した場合は、Secretsの設定とワークフローファイルの内容を再確認

## 5. トラブルシューting

### よくある問題と解決方法

#### 403 Forbidden エラー
- **原因**: Personal Access Tokenの権限不足
- **解決方法**: PATに`packages:write`権限が含まれていることを確認

#### Secrets が見つからないエラー
- **原因**: Secret名の不一致またはSecret未設定
- **解決方法**: リポジトリのSecrets設定を確認し、正確な名前で設定されているか確認

#### ワークフローが実行されない
- **原因**: GitHub Actionsが無効になっている
- **解決方法**: リポジトリ設定でGitHub Actionsを有効化

## 6. カスタマイズ

### 6.1 実行タイミングの変更
プルリクエスト以外のタイミングで実行したい場合は、`on:` セクションを修正：

```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    types: [opened, synchronize, reopened]
```

### 6.2 実行条件の追加
特定のファイルが変更された場合のみ実行したい場合：

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened]
    paths:
      - '**.py'
      - '**.js'
      - '**.ts'
```

## サポート

問題が発生した場合は、組織の管理者に連絡するか、GitHubのActionsログを確認して詳細なエラー情報を取得してください。