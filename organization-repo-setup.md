# 組織リポジトリセットアップ手順

このドキュメントでは、CodeCopyrightChecker組織リポジトリを最初から作成し、必要なファイルをセットアップする手順を説明します。

## 著作権表示

Unpublished Copyright (c) 2025 NTT, Inc. All rights reserved.

## 1. GitHubリポジトリの作成

### 1.1 新しいリポジトリの作成
1. GitHub組織のページに移動
2. **New repository** をクリック
3. リポジトリ名: `CodeCopyrightChecker`
4. **Public** を選択（組織内の他のリポジトリから使用するため）
5. **Create repository** をクリック

### 1.2 基本設定
1. **Settings** > **Actions** > **General** に移動
2. **Actions permissions** で "Allow all actions and reusable workflows" を選択
3. **Workflow permissions** で "Read and write permissions" を選択

## 2. ディレクトリ構造の作成

以下のディレクトリ構造を作成：

```
CodeCopyrightChecker/
├── action.yml                 # GitHub Actionの定義ファイル
├── README.md                  # プロジェクトの説明
├── requirements.txt           # Python依存関係
├── docker/
│   └── Dockerfile            # Dockerイメージ定義
└── scripts/
    └── gitDiffCheck.py       # メインスクリプト
```

## 3. 各ファイルの作成

### 3.1 action.yml の作成
リポジトリのルートに `action.yml` をアップロード

### 3.2 docker/Dockerfile の作成
`docker/Dockerfile` をアップロード

### 3.3 requirements.txt の作成
リポジトリのルートに `requirements.txt` をアップロード

### 3.4 README.md の作成
`README.md` を作成：

```markdown
# CodeCopyrightChecker

GitHub Actions用のコンポジットアクションで、プルリクエストの変更内容に対して著作権侵害チェックを実行します。

## 機能

- プルリクエストの差分を自動的に取得
- Azure Content Safety APIを使用した著作権侵害検出
- HuggingFaceモデルを使用した高度な分析
- 結果をプルリクエストにコメントとして投稿

## 使用方法

### 基本的な使用方法

```yaml
- name: Run copyright check
  uses: YOUR_ORG_NAME/CodeCopyrightChecker@main
  with:
    pr_number: ${{ github.event.pull_request.number }}
    GH_TOKEN: ${{ secrets.GH_TOKEN }}
    HUGGINGFACE_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
    AACS: ${{ secrets.AACS }}
    AACSAPIkey: ${{ secrets.AACSAPIkey }}
```

### 入力パラメータ

| パラメータ | 説明 | 必須 | デフォルト |
|------------|------|------|------------|
| `pr_number` | プルリクエスト番号 | Yes | - |
| `python_version` | 使用するPythonバージョン | No | "3.11" |
| `GH_TOKEN` | GitHub APIアクセス用トークン | Yes | - |
| `HUGGINGFACE_TOKEN` | HuggingFace APIトークン | Yes | - |
| `AACS` | Azure Content Safety エンドポイントURL | Yes | - |
| `AACSAPIkey` | Azure Content Safety APIキー | Yes | - |

## セットアップ

詳細なセットアップ手順については、以下のドキュメントを参照してください：

- [組織リポジトリセットアップ手順](organization-repo-setup.md) : 本ドキュメント
- [ユーザーリポジトリセットアップ手順](user-repo-setup.md)

### 3.5 scripts/gitDiffCheck.py の作成
`scripts/gitDiffCheck.py` を作成（既存のスクリプトをここに配置）

## 4. GitHub Container Registry の設定

### 4.1 パッケージ権限の設定
1. 組織の **Settings** > **Packages** に移動
2. **Package creation** で必要に応じて適切な権限を設定
3. **Package visibility** で必要に応じて設定

## 5. テスト用リポジトリでの動作確認

### 5.1 テストリポジトリの作成
1. 組織内に新しいテストリポジトリを作成
2. [ユーザーリポジトリセットアップ手順](user-repo-setup.md)に従ってワークフローを設定
3. テスト用のプルリクエストを作成

### 5.2 動作確認
1. プルリクエストを作成
2. GitHub Actionsが正常に実行されることを確認
3. Dockerイメージがビルド・プッシュされることを確認
4. スクリプトが正常に実行されることを確認

## 6. 本番環境への展開

### 6.1 タグの作成
安定版をリリースする際は、適切なタグを作成：

```bash
git tag -a v1.0.0 -m "First stable release"
git push origin v1.0.0
```

### 6.2 ドキュメントの更新
- README.mdの更新
- 使用例の提供
- トラブルシューティングガイドの作成

## 7. メンテナンス

### 7.1 定期的な更新
- 依存関係のアップデート
- セキュリティパッチの適用
- 新機能の追加

### 7.2 監視
- GitHub Container Registryの使用量監視
- アクションの実行ログ監視
- エラー率の追跡

## トラブルシューティング

### よくある問題

#### Dockerイメージのビルドエラー
- **原因**: Dockerfileの構文エラーまたは依存関係の問題
- **解決方法**: ローカルでDockerイメージをビルドしてテスト

#### 権限エラー
- **原因**: GitHub ActionsまたはGHCRの権限設定不備
- **解決方法**: 組織とリポジトリレベルでの権限設定を確認

#### スクリプト実行エラー
- **原因**: 環境変数の未設定またはAPIキーの問題
- **解決方法**: Secretsの設定とAPIキーの有効性を確認

## サポート

問題が発生した場合は、以下の情報を含めてIssueを作成してください：

- エラーメッセージ
- 実行ログ
- 使用している設定
- 期待される動作と実際の動作
