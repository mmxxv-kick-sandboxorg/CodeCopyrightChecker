# 組織リポジトリセットアップ手順

このドキュメントでは、CodeCopyrightChecker組織リポジトリを最初から作成し、必要なファイルをセットアップする手順を説明します。

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
リポジトリのルートに `action.yml` を作成：

```yaml
# CodeCopyRightChecker/action.yml
name: 'Copyright Check'
description: 'Check copyright infringement in pull requests using Azure Content Safety API'

inputs:
  pr_number:
    description: 'Pull request number to check'
    required: true
  python_version:
    description: 'Python version to use'
    required: false
    default: "3.11" 
  GH_TOKEN:
    description: 'GitHub token for API access'
    required: true
  HUGGINGFACE_TOKEN:
    description: 'HuggingFace access token'
    required: true
  AACS:
    description: 'Azure Content Safety endpoint URL'
    required: true
  AACSAPIkey:
    description: 'Azure Content Safety API key'
    required: true

runs:
  using: "composite" 
  steps:
    # 呼び出し元のリポジトリをチェックアウト（PR差分取得のため）
    - name: Checkout caller repository
      uses: actions/checkout@v4

    # Set up Docker environment
    - name: "2. enable docker buildx"
      uses: docker/setup-buildx-action@v3

    # Log in GitHub Container Registry
    - name: "3. login ghcr.io"
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ inputs.GH_TOKEN }}

    - name: "4. generate metadata"
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ghcr.io/${{ github.actor }}/ghcr-container-push-demo
        tags: |
          type=raw,value=latest
          type=sha,prefix=,suffix=,format=short

    - name: "5. build container image, and push"
      uses: docker/build-push-action@v5
      with:
        context: ${{ github.action_path }}
        file: ${{ github.action_path }}/docker/Dockerfile
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=registry,ref=ghcr.io/${{ github.actor }}/ghcr-ccc-container:buildcache
        cache-to: type=registry,ref=ghcr.io/${{ github.actor }}/ghcr-ccc-container:buildcache,mode=max

    - name: "6. run gitDiffCheck.py with PR number"
      shell: bash
      env:
        GH_TOKEN: ${{ inputs.GH_TOKEN }}
        HUGGINGFACE_TOKEN: ${{ inputs.HUGGINGFACE_TOKEN }}
        AACS: ${{ inputs.AACS }}
        AACSAPIkey: ${{ inputs.AACSAPIkey }}
        PYTHONIOENCODING: utf-8
      run: |
         docker run --rm \
           -e GITHUB_TOKEN=${{ inputs.GH_TOKEN }} \
           -e AACS=${{ inputs.AACS }} \
           -e AACSAPIkey=${{ inputs.AACSAPIKEY }} \
           -e HUGGINGFACE_TOKEN=${{ inputs.HUGGINGFACE_TOKEN }} \
           -v ${{ github.workspace }}:/app \
           -w /app \
           ghcr.io/${{ github.actor }}/ghcr-container-push-demo:latest \
           bash -c "git config --global --add safe.directory /app && python gitDiffCheck.py ${{ inputs.pr_number }}"
```

### 3.2 docker/Dockerfile の作成
`docker/Dockerfile` を作成：

```dockerfile
# Dockerfile
FROM ubuntu:22.04

# required package install
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl gh && \
    rm -rf /var/lib/apt/lists/*

# Hugging Face Transformers and dependency package install
RUN pip3 install --no-cache-dir transformers torch accelerate

# work directory
WORKDIR /app

# copy check script to container
COPY scripts/gitDiffCheck.py /app/gitDiffCheck.py

CMD ["python3", "gitDiffCheck.py"]
```

### 3.3 requirements.txt の作成
`requirements.txt` を作成：

```txt
transformers>=4.21.0
torch>=1.12.0
accelerate>=0.12.0
requests>=2.28.0
PyGithub>=1.55
azure-ai-contentsafety>=1.0.0
```

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

- [組織リポジトリセットアップ手順](organization-repo-setup.md)
- [ユーザーリポジトリセットアップ手順](user-repo-setup.md)

## ライセンス

[MIT License](LICENSE)
```

### 3.5 scripts/gitDiffCheck.py の作成
`scripts/gitDiffCheck.py` を作成（既存のスクリプトをここに配置）

## 4. GitHub Container Registry の設定

### 4.1 パッケージ権限の設定
1. 組織の **Settings** > **Packages** に移動
2. **Package creation** で適切な権限を設定
3. **Package visibility** で必要に応じて設定

### 4.2 Personal Access Token の作成
組織レベルでPATを作成する場合：
1. 組織の **Settings** > **Developer settings** > **Personal access tokens**
2. **Fine-grained tokens** または **Tokens (classic)** を選択
3. 必要な権限を設定：
   - `packages:write`
   - `packages:read`
   - `contents:read`

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
- トラブルシューtiングガイドの作成

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

## 貢献

プロジェクトへの貢献を歓迎します。プルリクエストを送信する前に、以下を確認してください：

- コードスタイルの準拠
- テストの実行
- ドキュメントの更新