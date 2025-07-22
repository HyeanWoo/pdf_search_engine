# 문서 기반 질의응답 시스템

이 프로젝트는 PDF 문서를 파싱하고, OpenAI API와 Faiss를 이용하여 문서 내용에 기반한 답변을 출력하는 시스템입니다.

## 주요 기능

- PDF 문서 파싱: PDF 파일의 텍스트를 추출합니다.
- 텍스트 임베딩 및 벡터 검색: OpenAI Embedding API와 Faiss를 사용하여 벡터 기반 문서에서 관련 정보를 검색합니다.
- 문서 기반 답변 생성: OpenAI chat API를 사용하여 검색된 내용을 기반으로 답변을 생성합니다.
- 캐싱 기능: 임베딩 처리한 문서의 인덱스를 저장하여 불필요한 API 호출을 방지합니다.

## 실행 방법

#### 사전 요구사항

- Python 3.8 이상

#### 1. 프로젝트 클론

```bash
git clone https://github.com/HyeanWoo/pdf_search_engine
cd pdf_search_engine
```

#### 2. 가상 환경 생성 및 활성화

```bash
conda create -n pdf-search-engine python=3.10
conda activate pdf-search-engine
```

#### 3. 라이브러리 설치

```bash
pip install -r requirements.txt
```

#### 4. 환경 변수 설정

1. 프로젝트 루트 폴더에 `.env` 파일을 생성합니다.
2. `OPENAI_API_KEY`항목을 만들고 본인의 OpenAI API를 입력합니다.

```
OPENAI_API_KEY={your OpenAI API key}
```

#### 5. 실행

아래 명령어를 입력하면 프로젝트가 실행됩니다.

```bash
python main.py
```

## 구조 설명

```python
doc-qa-project/
├── cache/                  # 인덱스와 텍스트 청크 파일 저장
├── data/                   # PDF 문서 보관
├── .env                    # OpenAI API 키 등 환경변수 관리
├── config.py               # 프로젝트 주요 설정값 관리
├── document_processor.py   # PDF 파싱과 텍스트 처리 담당
├── gpt_handler.py          # OpenAI API(임베딩, 답변 생성) 호출
├── main.py                 # 메인 실행 로직
├── prompt.py               # 프롬프트 템플릿 관리
├── vector_retriever.py     # FAISS 인덱스 생성, 저장, 로드 및 벡터 기반 정보 검색 로직
├── requirements.txt        # 실행에 필요한 라이브러리 목록
└── README.md               # 프로젝트 설명서
```

## 설계 의도

이 프로젝트는 RAG(Retrieval-Augmented Generation)를 기반으로 설계되었습니다. RAG는 LLM의 문제점인 환각, 최신 정보 부족, 비공개 데이터 접근 제한 등을 해결할 수 있습니다.

- 사용자 질문과 관련된 내용을 기반으로 답변을 생성하여 신뢰도를 높입니다.
- 전체 컨텍스트를 넣지않고 벡터 검색을 통해 가장 연관되어 있는 내용을 전달하여 불필요한 정보를 방지해 비용과 환각 현상을 줄일 수 있습니다.
- 각 파이썬 파일은 기능별로 유지보수와 확장성을 고려하여 분리했습니다.

#### 프롬프트 전략

- LLM에게 'PDF 문서 검색 엔진'이라는 구체적인 역할을 부여해 초기 방향을 설정합니다.
- 입력 형식과 답변 형식을 명확히 제시하여 일관된 답변을 생성하도록 유도합니다.
- 핵심적인 제약사항을 부여하여 부정확한 답변을 방지합니다.
- 구체적인 예시를 제공하여 원하는 답변을 LLM이 제공하도록 학습합니다.

#### 텍스트 처리 전략

- 벡터 검색과 맥락 파악에서 의미가 끊기는 문제를 방지하기 위해 단어 단위로 텍스트를 나누었습니다.
- `chunk_size`는 PDF 문서의 페이지당 평균값에 가까운 256으로 설정했습니다.
- 맥락이 누락되는 것을 방지하기위해 15% 정도의 `overlap`을 두었습니다.
