# Unpublished Copyright (c) 2025 NTT, Inc. All rights reserved.
# 
# Copyright Infringement Detection tool for ubuntu
# 

# import section
import sys
import json
import time
import subprocess
import requests
import argparse
import difflib
import urllib.request
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# define config param.
config = {
    "k": 0.5,
    "thr": -0.012
}

# 実行時間の計測開始
start_time = time.time()

# GitHub のコメント先頭に付与する固定文字列
strGitCommentHeader = '[Copyright Infringement Detection] '

# REST API endpoint
url = os.getenv("AACS")
# AACS APIkey
AACSAPIkey = os.getenv("AACSAPIkey")

def load_model(name):
    """
    Load a pre-trained language model and tokenizer from local directory or Hugging Face.
    
    Args:
        name (str): Model name/path from Hugging Face Hub or local directory path
        
    Returns:
        tuple: (model, tokenizer) - Loaded model and tokenizer
        
    Raises:
        RuntimeError: If HUGGINGFACE_TOKEN environment variable is not set for remote models
        FileNotFoundError: If local model directory is not found
    """
    # Check if the name is a local path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(script_dir, name)
    
    # Check if it's a local directory
    if os.path.isdir(local_model_path):
        print(f"Loading local model from: {local_model_path}")
        
        # Load local model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(local_model_path, return_dict=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        # Load from Hugging Face Hub
        print(f"Loading model from Hugging Face Hub: {name}")
        
        # Read the login token for Hugging Face from an environment variable.
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise RuntimeError("Login token for Hugging Face is not set in environment variables.")
            
        # Load other models (e.g., Qwen) using AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(name, return_dict=True, device_map='auto', token=token)
        tokenizer = AutoTokenizer.from_pretrained(name)

    # Inference mode. Training is disabled.
    model.eval()

    return model, tokenizer

def getTokenLogProbs(code, model, tokenizer, add_special_tokens=False):
    '''
    Get the log probability of each token in the given code.
    
    This function tokenizes the input code and calculates the log probability
    for each token using the loaded language model.
    
    Args:
        code (str): Source code to analyze
        model: Loaded language model
        tokenizer: Tokenizer for the model
        add_special_tokens (bool): Whether to add special tokens during tokenization
        
    Returns:
        torch.Tensor: Log probabilities for each token in the code
    '''

    # Tokenize the input code
    input_ids = tokenizer.encode(code, add_special_tokens=add_special_tokens)
    input_ids = torch.tensor([input_ids])  # [1, seq_len]
    input_ids = input_ids.to(model.device)

    # Retrieve logits with gradients disabled (inference only).
    with torch.no_grad():
        logits = model(input_ids).logits  # [1, seq_len, vocab]

    # Shift logits and labels for next-token prediction
    logits = logits[:, :-1, :]  # Remove last token from logits
    labels = input_ids[:, 1:]   # Remove first token from labels

    # Get log-probability of each token.
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.squeeze(0)

def calculateMaxKPercentProb(model, tokenizer, code, k):
    """
    Calculate the MAX-K% probability score for the given code.
    
    This function computes a metric that represents the average log probability
    of the top K% most likely tokens in the code. This metric is used to
    detect commonplace expression or protectable expression.
    
    Args:
        model: Loaded language model
        tokenizer: Tokenizer for the model
        code (str): Source code to analyze
        k (float): Percentage of tokens to consider (0.0 to 1.0)
        
    Returns:
        float: MAX-K% probability score
    """
    # Get the log probability of each token in the given code.
    token_log_probs = getTokenLogProbs(code, model, tokenizer)

    # Take the top K% log probabilities
    k_length = int(len(token_log_probs) * k)
    max_k_percent_token_log_probs = np.sort(token_log_probs)[k_length:]

    # Calculate the MAX-K% PROB - average of top K% token probabilities
    max_k_percent_prob = np.mean(max_k_percent_token_log_probs).item()

    return max_k_percent_prob


def callGitHubCLIdiff(strCmd):
    """GitHub CLIを呼び出し、結果を取得するラッパー関数

    Args:
        strCmd: 'gh pr diff ' + プルリクエストID

    Returns:
        result: 'before', 'after', 'diff' の要素を持つ辞書のリスト
    """
    # GitHub CLIの呼び出し
    try:
        procObj = subprocess.run(strCmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if procObj.returncode != 0:
            print('ERROR : undefined result has returned from GitHub CLI(diff).', procObj.stderr)
            exit(1)
    except:
        print('ERROR has occured executing GitHub CLI(diff).', e)
        exit(1)

    # 結果を格納するリストの初期化
    result = []

    # 行ごとに処理
    lines = procObj.stdout.splitlines()
    current_diff = None
    line_number = 0  # 行番号カウンタ

    # ルール
    # 'diff --git 'で始まる場合、半角スペースで区切られた続く要素を"before" "after"として取得
    # '@@' で始まる場合、'\ No newline at end of file' という行が出現するまで以降の行について要素"diff" として取得
    # 'diff' の要素は複数行(改行)を許容する
    # 'diff' の要素で行の先頭文字が '+' の場合、その'+'の文字を取り除く
    # 'diff' の要素で行の先頭文字が '+' ではない場合、その行を 'diff' 要素から取り除く
    for line in lines:
        # 'diff --git 'で始まる行を処理
        if line.startswith('diff --git '):
            if current_diff is not None:
                result.append(current_diff)
            parts = line.split()
            current_diff = {
                "before": parts[2].split("/")[-1],
                "after": parts[3].split("/")[-1],
                "diff": []
            }
            line_number = 0  # 行番号をリセット
        # '@@'で始まる行を処理
        elif line.startswith('@@'):
            # 増分行算出のための数値を取得
            # 例: @@ -8,6 +8,27 @@ から +8 の部分を抽出
            import re
            match = re.search(r'\+(\d+)', line)
            if match:
                line_number = int(match.group(1))
        # '\ No newline at end of file'が出現した場合、現在のdiffを確定
        elif line == '\\ No newline at end of file':
            if current_diff is not None:
                # diff要素を連結して保存
                result.append(current_diff)
                current_diff = None
        # '+'で始まる行を処理
        # '-'および' 'で始まる行は無視
        elif current_diff is not None:
            if line.startswith('+++') or line.startswith('---'):
                continue  # '+++'または'---'で始まる行は無視
            elif line.startswith('+'):
                current_diff["diff"].append({
                    "content": line[1:],  # '+'を取り除く
                    "line_number": line_number
                })
                line_number += 1  # '+'行の場合は行番号をインクリメント
            elif line.startswith(' '):
                line_number += 1  # ' '行の場合も行番号をインクリメント
            elif line.startswith('-'):
                # '-'行は行番号をインクリメントしない（削除行なので）
                pass

    # 最後のdiffが残っている場合は結果に追加
    if current_diff is not None:
        result.append(current_diff)

    return result

def callAzureCLI(data):
    """Azure Content Safety APIを呼び出し、結果を取得するラッパー関数

    Args:
        data: コードスニペット

    Returns:
        response.text: APIが返すJSON形式の文字列
    """
    # リクエストヘッダを指定
    headers = {
        'Ocp-Apim-Subscription-Key': AACSAPIkey,
        'Content-Type': 'application/json'
    }

    # JSON形式でデータを送信する
    payload = {
        "code": data
    }

    # POSTリクエストを送信
    # HTTP例外をキャッチした場合はexcept節で処理
    try:
        response = requests.post(url, headers=headers, json=payload)
    except requests.exceptions.RequestException as e:
        print('    Error has occured:', e)
        return '{"error":{"code":"RequestException","message":"'+ str(e) + '","details":[]}}'

    # レスポンスの確認
    # ToDo : HTTPステータスコードに応じた処理を追加
    if response.status_code == 200:
        # no operation. comment out for debug
        pass
        # 200 OKの例
        # print('Result :', response.text)
    elif response.status_code == 400:
        # response.textをJSONとして解析して、エラーメッセージを抽出
        response_json = response.json()
        if 'error' in response_json and 'message' in response_json['error'] and response_json['error']['code'] == 'InvalidRequestBody':
            error_message = response_json['error']['message']
            print('Bad Request :', error_message)
        else:
            # それ以外の400エラーの場合、status_codeとresponse.textを表示する
            print('Bad Request :', response.status_code, response.text)
            # ToDo: gh pr comment でエラーメッセージをプルリクエストにコメントとして追加する
    else:
        print('Error has occured:', response.status_code, response.text)
    
    return response.text

def callGitHubCLIcomment(prNumber, comment):
    """GitHub CLIを呼び出し、プルリクエストにコメントするラッパー関数

    Args:
        prNumber: プルリクエスト番号
        comment: コメント本文

    Returns:
        result: boolean
    """
    # GitHub CLIの呼び出し
    # gh pr comment <プルリクエスト番号> --body "<コメント本文>"
    strCmd = ['gh', 'pr', 'comment', prNumber, '--body', '"' + comment + '"']
    print('Command:', strCmd)
    try:
        # shell=Falseがデフォルトなので明示的に指定しても良い
        procObj = subprocess.run(strCmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if procObj.returncode != 0:
            print('ERROR : undefined result has returned from GitHub CLI(comment).', procObj.stderr)
            return False
        return True
    except:
        print('ERROR : could not execute GitHub CLI(comment).')
        return False

    return True

def matchSnippetCode(fileUri, diffBody, diffElements):
    """差分のコードスニペットとリポジトリ上のファイルとの一致部分を探す

    Args:
        fileUri: ファイル名(URI)
        diffBody: 差分のコードスニペット
        diffElements: diff要素のリスト（content と line_number を含む辞書のリスト）

    Returns:
        matchSnippet: 一致したコードスニペット（見つからなかった場合は空リスト）
    """
    try:
        # proxy_handler = urllib.request.ProxyHandler()  # 環境変数を参照
        # opener = urllib.request.build_opener(proxy_handler)
        # urllib.request.install_opener(opener)

        # URIからテキストを取得
        with urllib.request.urlopen(fileUri) as response:
            remote_text = response.read().decode('utf-8')

        diff = difflib.Differ()

        # リポジトリ上のファイルと差分のスニペットをそれぞれ行単位で分割したリストにする
        a_lines = remote_text.splitlines(keepends=False)
        b_lines = diffBody.splitlines(keepends=False)
        difflist = list(diff.compare(a_lines, b_lines))

        # line_numberのマッピングを作成（従来の文字列形式との互換性を保つ）
        line_number_map = {}
        if diffElements and isinstance(diffElements[0], dict):
            for i, elem in enumerate(diffElements):
                line_number_map[i] = elem['line_number']

        first_match_index = None
        last_match_index = None
        matched_line_numbers = []  # マッチした行番号を記録

        b_line_num = 0  # b_linesの行番号カウンタ

        output_diff = diff.compare(a_lines, b_lines)

    except FileNotFoundError:
        print(f'File not found: {fileUri}')
        return ""
    except Exception as e:
        print(f'Error get file {fileUri}: {e}')
        return ""    

    matchSnippet = {"match": []}
    # 一致部分のソースコードを途中が一致しなくてももれなく抽出
    for line in difflist:
        # print(line)
        code = line[:2]
        text = line[2:]

        if code == '  ':  # 一致行
            if first_match_index is None:
                first_match_index = b_line_num
            
            # line_number情報を取得
            line_number = line_number_map.get(b_line_num, b_line_num + 1)  # デフォルトは1ベースの行番号
            matched_line_numbers.append(line_number)
            
            #print( f"match : %s ", line)
            matchSnippet["match"].append({
                "content": text,
                "line_number": line_number
            })
            last_match_index = b_line_num
            b_line_num += 1
        elif code == '- ':  # aにだけある行
            # b_line_numは増やさない
            pass
        elif code == '+ ':  # bにだけある行
            if first_match_index is None:
                pass
            else:
                # line_number情報を取得
                line_number = line_number_map.get(b_line_num, b_line_num + 1)  # デフォルトは1ベースの行番号
                matched_line_numbers.append(line_number)
                
                matchSnippet["match"].append({
                    "content": text,
                    "line_number": line_number
                })
            b_line_num += 1
        elif code == '? ':
            # 差分の詳細行は無視
            pass

    # 一致行が全くなかった場合は空リストを返す
    if first_match_index is None:
        print("No match repository and snippet.")
        return None, None, []

    # マッチした行番号の最小値と最大値を計算
    if matched_line_numbers:
        first_match_line_number = min(matched_line_numbers)
        last_match_line_number = max(matched_line_numbers)
    else:
        first_match_line_number = first_match_index
        last_match_line_number = last_match_index

    return first_match_line_number, last_match_line_number, matchSnippet


# --- main ---
def main(args):
    """メイン関数
        チェック結果をJSON形式で表示する
    Args:
        args: コマンドライン引数（スクリプト名を除く）

    Returns:
        なし

    """
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='publish GitHub Pull-request diff command, and Query to a REST API.')
    parser.add_argument('pullId', type=str, help='Number of Pull-request ID')
    args = parser.parse_args()

    # pullIdの取得
    pullId = args.pullId
    
    # 処理対象の文字列取得
    #strCmd = 'gh pr diff ' + pullId # for Windows
    strCmd = ['gh', 'pr', 'diff', pullId] # for ubuntu

    print(strCmd)

    # GitHub CLIの呼び出し
    strResult = callGitHubCLIdiff(strCmd)
    print('GitHub CLI Result:', strResult)
    # 存在しないプルリクエスト番号を指定して strResultが空のリストになった場合は終了
    if not strResult:
        print('ERROR:指定された差分が取得できません。Pull-request IDを確認してください')
        exit(1)

    model, tokenizer = load_model("Qwen/Qwen2.5-Coder-0.5B")

    # 結果情報の初期化
    all_diffs = []
    # strResultに対して1件ずつ処理
    for item in strResult:
        dataset = item['diff']
        # diff要素から文字列を抽出（content部分のみ）
        if dataset and isinstance(dataset[0], dict):
            data = '\n'.join([d['content'] for d in dataset])  # 辞書形式の場合
        else:
            data = ''.join(dataset)  # 従来の文字列形式の場合（後方互換性）
        if data == "":
            # 空のdiff(削除だけの差分)の場合、APIを呼び出さずにスキップ
            item.update(
                {
                    "detected": False,
                    "license": "",
                    "sourceUrls": []
                }
            )
            print("  Skip empty diff for file", item['after'])
            continue   # 空のdiffはスキップ

        # Azure CLIの呼び出し
        strAzureResult = callAzureCLI(data)
        # JSONのtrueはPythonのTrueに変換される必要があるため、json.loadsで読み込む
        jsonAzureResult = json.loads(strAzureResult)

        # 結果をitemに追加
        # 'protectedMaterialAnalysis'キーが存在し、かつその中に'codeCitations'キーが存在し、さらに'codeCitations'リストが空でない場合
        if 'protectedMaterialAnalysis' in jsonAzureResult and 'codeCitations' in jsonAzureResult['protectedMaterialAnalysis'] and len(jsonAzureResult['protectedMaterialAnalysis']['codeCitations']) > 0:
            # チェック結果が true
            print('  Similar public code snippets were detected in file', item['after'])

            # Calculate max_k_percent_prob using the loaded model and configuration
            max_k_percent_prob = calculateMaxKPercentProb(model, tokenizer, data, config["k"])
            
            # Determine prediction result based on threshold comparison
            # If max_k_percent_prob >= threshold, return True (likely Commonplace Expression)
            # If max_k_percent_prob < threshold, return False (likely Protectable Expression)
            is_commonplace_expression = max_k_percent_prob >= config["thr"]

            # matchSnippetCodeを先に実行して行番号情報を取得
            # diff要素から文字列を抽出（content部分のみ）
            if dataset and isinstance(dataset[0], dict):
                testSnippet = '\n'.join([d['content'] for d in dataset])  # 辞書形式の場合
            else:
                testSnippet = '\n'.join(dataset)  # 従来の文字列形式の場合（後方互換性）

            matchCode = None
            first_match_line_number = None
            last_match_line_number = None
            
            # 'codeCitations'が存在し、かつ空でない場合、'detected', 'license', 'sourceUrls'の値を追加
            item.update(
                {
                    "detected": jsonAzureResult['protectedMaterialAnalysis']['detected'],
                    "license": jsonAzureResult['protectedMaterialAnalysis']['codeCitations'][0]['license'] if jsonAzureResult['protectedMaterialAnalysis']['codeCitations'] else "",
                    "sourceUrls": jsonAzureResult['protectedMaterialAnalysis']['codeCitations'][0]['sourceUrls'] if jsonAzureResult['protectedMaterialAnalysis']['codeCitations'] else [],
                    "is_commonplace_expression": is_commonplace_expression,            # True if likely AI-generated, False if likely human-written
                    "max_k_percent_prob": max_k_percent_prob  # The calculated MAX-K% probability score
                }
            )

            for originUri in item['sourceUrls']:
                # Convert originUri to rawUri
                targetUri = originUri.replace("github.com/", "raw.githubusercontent.com/").replace("/tree/", "/").replace("%2F", "/")

                result = matchSnippetCode(targetUri, testSnippet, dataset)

                # If 404 occurs, matchSnippetCode returns an empty string "", so skip to the next URL
                if result == "":
                    continue

                # If result is None or an empty list, also skip to the next URL
                if result is None or result == []:
                    continue

                # Otherwise, result is expected to be a tuple (stline, edline, matchCode)
                stline, edline, matchCode = result
                if matchCode:
                    # If a match is found, break the loop
                    first_match_line_number = stline
                    last_match_line_number = edline
                    break

            # If matchCode is None or an empty list, set matchSnippet to an empty list
            if matchCode is None or matchCode == []:
                item.update({"matchSnippet": []})
            else:
                item.update({"matchSnippet": matchCode["match"]})

            # callGitHubCLIcomment でプルリクエストにコメントを追加
            # 行番号情報を含めたコメントを作成
            commentPartCR = " "
            line_info = ""
            if first_match_line_number is not None and last_match_line_number is not None:
                if first_match_line_number == last_match_line_number:
                    line_info = f" (Line: {first_match_line_number})"
                else:
                    line_info = f" (Lines: {first_match_line_number}-{last_match_line_number})"
            
            if is_commonplace_expression:
                #comment = strGitCommentHeader + f"Even though similar public code snippet was detected in file {item['after']}, it was COMMONPLACE EXPRESSION!!!" + commentPartCR + f"License: {item['license']}." + commentPartCR + f"Source URLs: {', '.join(item['sourceUrls'])}"
                comment = strGitCommentHeader + f"Similar public code snippet was detected in file {item['after']}{line_info}." + commentPartCR + f"   License: {item['license']}." + commentPartCR + f"   [COMMONPLACE EXPRESSION] MAX-K% PROB = {max_k_percent_prob}   Thr = {config['thr']}"
            else:
                #comment = strGitCommentHeader + f"Similar public code snippet was detected in file {item['after']}." + commentPartCR + f"License: {item['license']}." + commentPartCR + f"Source URLs: {', '.join(item['sourceUrls'])}"
                comment = strGitCommentHeader + f"Similar public code snippet was detected in file {item['after']}{line_info}." + commentPartCR + f"   License: {item['license']}." + commentPartCR + f"   [COPYRIGHTABLE EXPRESSION] MAX-K% PROB = {max_k_percent_prob}   Thr = {config['thr']}"
            # print('Comment:', comment)
            if callGitHubCLIcomment(pullId, comment):
                print('  Comment added to Pull-request ID', pullId)

        else:
            # チェック結果が false
            # APIでエラーの場合はその旨コメントする
            if 'error' in jsonAzureResult and jsonAzureResult['error']['code'] != 'InvalidRequestBody':
                error_message = jsonAzureResult['error'].get('message', 'Unknown error')
                comment = strGitCommentHeader + f"Bad API Request : {error_message}."
                if callGitHubCLIcomment(pullId, comment):
                    print('  Comment added to Pull-request ID', pullId)

            # 'codeCitations'が存在しない場合、'detected'の値のみ追加し、他は空に設定
            item.update(
                {
                    "detected": jsonAzureResult.get('protectedMaterialAnalysis', {}).get('detected', False),
                    "license": "",
                    "sourceUrls": []
                }
            )

    # output JSON format
    print(strResult)
    print('check completed.')

    # 実行時間の計測終了
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Elapsed time: {execution_time:.2f}sec.')

# --- wrapper ---
if __name__ == "__main__":

    main(sys.argv[1:])  # command line params exclude command name
