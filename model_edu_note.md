# word Encoding
위치 임베딩은 토큰 임베딩과 크기가 같아야 함. (행렬 합을 해야 하기 때문에)
절대 위치 임베딩은 잘 이해 안됨. (sinusoidal positional encoding)
상대 위치 임데빙이 스텝마다 변한다는 의미가 뭔지?
ROPE - 단어를 회전 하낟고 하는데. 그 의미가 무엇인지??

# Token
Special Token : 특수한 문장의 의미등을 넣어서 활용하는 용도.
Self-supervised dataset 은 한단어 다음을 타킷으로 삼아서 생성하는 방식임.

# attention
문맥을 활용하여 뒤에 올 단어에 따른 확률 값을 더하는 것임. (반영할 확률값은 어디서?? => softmax 의 결과값)
self-attention 은 나 자신과 비교하는 것임. 
## Q,K,V
Q (query) : 질문을 던지는 주체
K (Key) : 질문에 답하는 대상
V (Value) : 전달할 정보

Q, K를 분리하지 않으면 Attention이 제대로 되지 않음
자기 자신이 항상 내적의 최대값이 되고, 순서 변경해도 이를 인지 할기 어려움

K,V를 분리하지 않으면 의미가 섞일 수 있음
Key 는 어디를 볼지 를 정하는 검색용 단어
Value 는 무엇을 가져올지 정하는 실제 정보

## masking
미래 토큰을 알수 없음으로 현재까지의 토큰으로만 Attention score 를 계산하기 위함

## Multi-head Attention 
?? 이해 필요, 각각의 관점에서 보는 것이라는 데 잘 이해 안됨
따로 공부 필요. 

# 지식의 저장
FFN ?? 어떤 의미인지? 잘 이해 안됨.
새로운 Layer로 가면서 새로운 정보들이 만들어짐. (중첩적인 의미)


