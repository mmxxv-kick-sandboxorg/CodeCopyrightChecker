# ユーザーリポジトリセットアップ手順

このドキュメントでは、組織のCodeCopyrightCheckerアクションを使用するために、ユーザーリポジトリに必要な設定を行う手順を説明します。

## 著作権表示
Unpublished Copyright (c) 2025 NTT, Inc. All rights reserved.

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
`.github/workflows/checkinfringement.yml` ファイルを新規に作成し、資材の checkinfringement.yml の内容を設定

**注意**: `YOUR_ORG_NAME` を実際の組織名に変更してください。

## 3. Secrets の設定

### 3.1 Personal Access Token (PAT) の作成
1. GitHub の **Settings** > **Developer settings** > **Personal access tokens** > **tokens(classic)** に移動
2. **Generate new token(classic)** をクリック
3. 以下の権限を設定：
  Select scopes では下記にチェックを入れる
    - repo
    - write:packages
    - read:packages
    - delete:packages
    - read:org

4. トークンを生成し、安全な場所に保存

### 3.2 リポジトリSecrets の設定
リポジトリの **Settings** > **Secrets and variables** > **Actions** で以下のSecretsを追加：

| Secret名 | 説明 | 取得方法 |
|---------|------|----------|
| `GH_TOKEN` | GitHub Personal Access Token | 上記で作成したPAT |
| `HUGGINGFACE_TOKEN` | HuggingFace APIトークン | [HuggingFace Settings](https://huggingface.co/settings/tokens)で作成 |
| `AACS` | Azure Content Safety エンドポイントURL | Azure portalから取得した文字列 + "/contentsafety/text:detectProtectedMaterialForCode?api-version=2024-09-15-preview" |
| `AACSAPIkey` | Azure Content Safety APIキー | Azure portalから取得 |

## 4. テスト実行

### 4.1 プルリクエストの作成
1. 新しいブランチを作成
2. 何らかのファイルを変更
3. プルリクエストを作成
4. GitHub Actions が自動的に実行されることを確認

### 4.2 動作確認
- **Actions** タブでワークフローの実行状況を確認
- エラーが発生した場合は、Secretsの設定とワークフローファイルの内容を再確認

## 5. トラブルシューティング

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

### 6.1 実行条件の追加
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
