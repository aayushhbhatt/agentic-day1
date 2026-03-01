import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def main() -> None:
    # 1) Load environment variables
    load_dotenv()

    # 2) Read configuration from environment (safe defaults)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Create a .env file with OPENAI_API_KEY=... "
            "and ensure it is loaded via load_dotenv()."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    # 3) Initialize a LangChain chat model (required)
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )

    # ------------------------------------------------------------
    # 2️⃣ Context Break Demonstration (Naïve Invocation)
    # ------------------------------------------------------------
    print("\n=== Naïve string-based invocation (context breaks) ===")

    resp1 = llm.invoke("We are building an AI system for processing medical insurance claims.")
    print("\n[resp1]")
    print(resp1.content)

    resp2 = llm.invoke("What are the main risks in this system?")
    print("\n[resp2]")
    print(resp2.content)

    # Why the second question may fail or behave inconsistently:
    # Because `llm.invoke()` with a plain string is a stateless*call.
    # The model is not automatically provided the prior prompt ("medical insurance claims"),
    # so the second question ("this system") may be ambiguous. Thus the response provided by the model is generic and includes statement like it needs more context, anwsers could be:
    # - generically about "AI systems" instead of claims processing
    # - Guess the domain incorrectly
    # - Provide incomplete/irrelevant risk analysis
    # In production, this leads to inconsistent outputs and brittle UX unless the application
    # explicitly carries forward conversation state.

    # ------------------------------------------------------------
    # 3️⃣ Context Fix Using Messages API (structured history)
    # ------------------------------------------------------------
    print("\n=== Message-based invocation (context preserved) ===")

    messages = [
        SystemMessage(content="You are a senior AI architect reviewing production systems."),
        HumanMessage(content="We are building an AI system for processing medical insurance claims."),
        HumanMessage(content="What are the main risks in this system?"),
    ]

    resp3 = llm.invoke(messages)
    print("\n[resp3]")
    print(resp3.content)


if __name__ == "__main__":
    main()


"""
Reflection:

1. Why did string-based invocation fail?
    Each `invoke("...")` call is independent. Without explicitly re-supplying prior turns,
    the model does not know what "this system" refers to. The ambiguity causes generic,
    inconsistent, or incorrect answers depending on how the model guesses missing context.

2. Why does message-based invocation work?
    The messages list bundles the system role + the relevant user context + the follow-up
    question into a single request. This preserves conversational state by making the
    context explicit in the payload, so the model can reliably ground its answer in the
    medical insurance claims domain.

3. What would break in a production AI system if we ignore message history?
   User experience: follow-up questions become unreliable ("it forgets what I said").
   Safety/compliance: missing context can trigger unsafe or non-compliant responses
    (e.g., wrong domain assumptions, incorrect policy logic, poor medical/financial guidance).
   Correctness: workflows requiring state (multi-step forms, claim adjudication reasoning,
    ticket resolution) will degrade or fail.
   Observability/evals: outputs become non-deterministic and hard to debug because the
    effective input is incomplete and varies with model guesswork.
"""