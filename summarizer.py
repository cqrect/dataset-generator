import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASEURL = os.getenv("OPENAI_BASEURL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

open("summary.txt", "w+", encoding="utf-8").close()

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASEURL,
)

prompt = """"以下是一段文章内容，截取自王唯工教授的《看懂经气脉络》和常州金姆健康科技有限公司的介绍，包含在<article></article>标签内。现在，请你阅读、理解、整理、总结片段，生成文本摘要。摘要包含在<chunk></chunk>标签内。

注意以下几点：
1. 保证片段内容的完整性，不得丢失片段中的任何关键信息和知识点。
2. 如果片段中的信息包含你已知的信息，你可以尝试扩展片段内容使之更丰富、完整。
3. 片段可能是语义残缺的，请根据语境和上下文尝试补全，如果没有足够的信息补全就略过，避免错误的内容和无端的猜测。
4. 摘要要保证内容的完整性，保证摘要是可以单独阅读理解的片段，不要和上下文联系。比如不要出现“上文”、“前面提到过”、“根据之前所说的”、未表明身份的人称代词等。如果有，请明确说明指代的是什么。
5. 如果生成的摘要超过了500字，请分段，保证每段不超过500字，且相同的内容尽量集中在同一段。

格式如下：
<chunk>
摘要摘要...
</chunk>

或者：
<chunk>
摘要摘要（不超过500字）...
</chunk>
<chunk>
摘要摘要（不超过500字）...
</chunk>
...

注意chunk标签独占一行，每个chunk中只能有一段话且不超过500字。现在，请根据下面的片段按要求生成摘要：
<article>
{{article}}
</article>"""


def getSummary(chunk: str):
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "你是个善于总结摘要的智能助手"},
            {"role": "user", "content": prompt.replace("{{article}}", chunk)},
        ],
    )

    result = completion.choices[0].message.content.strip("\n")

    with open("summary.txt", "a+", encoding="utf-8") as wf:
        wf.write(result + "\n")

    print(result)


chunks = []
with open("output.txt", "r", encoding="utf-8") as f:
    context = ""
    chunkFlag = False
    for eachline in f.readlines():
        if eachline.startswith("<chunk>"):
            chunkFlag = True
            continue
        if eachline.startswith("</chunk>"):
            chunkFlag = False

        if chunkFlag:
            context += eachline
        if not chunkFlag:
            if context != "":
                chunks.append(context)

                if len(chunks) == 3:
                    content = "".join(chunks)
                    getSummary(content)
                    chunks.pop(0)

                context = ""

if len(chunks) > 0:
    content = "".join(chunks)
    getSummary(content)
