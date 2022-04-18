### < NNI 실행하기 >

1. 아래 명령어를 terminal에서 입력합니다. (각각 import lib과 nni를 다운받는 명령어 입니다.)

```bash
pip install importlib
python -m pip install --upgrade nni
```

- 만약 nni 다운로드에서 PyYAML 에러가 발생한다면 아래 명령어를 입력하고 nni를 다운받아 주세요.
    
    ```bash
    pip install --ignore-installed PyYAML
    ```
    

2. terminal에서 다음의 명령어를 입력하여 실행합니다.
    
    ```bash
    nnictl create --config nni/config.yml --port 30001
    ```
