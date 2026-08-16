import json
import re


# ---------------------------------------------------------------------------
# 1. SUMMARIZATION
# ---------------------------------------------------------------------------
# Evolution note: V1 (`SUMMARY_PROMPT_V1`) was a naive "Summarize this: {letter}"
# prompt with no role, no constraints, and no structure — it produced dense,
# hard-to-scan paragraphs and occasionally editorialized (e.g. describing a
# stated LACK of collateral as if it were a form of collateral). V2 fixes this
# by giving the model an explicit role and forcing a neutral, factual register
# via the system prompt, and by running at temperature=0 for consistency.

def SUMMARY_PROMPT_V2(letter):
    """Summarize a loan application letter, factually and neutrally."""
    return ask_llm(
        user_prompt=f"Summarize this loan application: {letter}",
        system_prompt="You are an assistant to a microfinance loan officer",
        temperature=0,
    )


# ---------------------------------------------------------------------------
# 2. STRUCTURED EXTRACTION
# ---------------------------------------------------------------------------
# Evolution note: built with an explicit JSON schema, a single worked few-shot
# example (using a letter NOT drawn from the evaluation set, to avoid leaking
# gold answers into the prompt), an explicit "use null, do not guess"
# instruction to prevent the model from inventing missing fields, and
# temperature=0 for reproducible, parseable output.

def EXTRACT_PROMPT(letter_text):
    """
    Build the (system_prompt, user_prompt) pair used to extract structured
    fields from a loan application letter.

    Returns a JSON object with exactly these keys:
        applicant_name (string), amount_ghs (number), purpose (string),
        monthly_profit_ghs (number or null), has_collateral_or_guarantor (boolean),
        repayment_months (number or null)
    """
    # Few-shot example letter (NOT one of the letters being evaluated)
    example_letter = """Dear Sir,
    My name is Victoria Brown Afari and I need GHS 4000 for my small nail tech business.
    I will use it to buy nail equipmemts. The profit I make each month is about GHS 1500.
    I have a guarantor. I can repay in 10 months.
    """

    example_json = {
        "applicant_name": "Victoria Brown Afari",
        "amount_ghs": 4000,
        "purpose": "small nail tech business",
        "monthly_profit_ghs": 1500,
        "has_collateral_or_guarantor": True,
        "repayment_months": 10,
    }

    system_prompt = (
        "You are a highly efficient data extraction bot. Your sole task is to extract specific information "
        "from loan application letters and return it as a JSON object. Adhere strictly to the following schema:\n"
        "- `applicant_name`: (string) The full name of the applicant.\n"
        "- `amount_ghs`: (number) The loan amount requested in Ghanaian Cedis.\n"
        "- `purpose`: (string) A brief description of what the loan is for.\n"
        "- `monthly_profit_ghs`: (number or null) The stated monthly profit of the applicant's business, if mentioned. Use null if not specified.\n"
        "- `has_collateral_or_guarantor`: (boolean) True if collateral or a guarantor is mentioned, False otherwise.\n"
        "- `repayment_months`: (number or null) The proposed repayment period in months, if mentioned. Use null if not specified.\n"
        "If a field is not stated in the letter, use null. Do not guess or invent details. "
        "Respond ONLY with the JSON object. Do not include any other text or formatting outside the JSON."
    )

    user_prompt = (
        f"Extract the following fields from the loan application letter and return them as a JSON object:\n"
        f"Letter:\n{example_letter}\n"
        f"JSON: {json.dumps(example_json)}\n\n"
        f"Letter:\n{letter_text}\n"
        f"JSON:"
    )

    return system_prompt, user_prompt


def extract_fields(letter_text):
    """
    Call the LLM with EXTRACT_PROMPT, strip any ```json fences, parse the
    result, and return a dict. Returns None (with a printed warning) if the
    response can't be parsed as valid JSON.
    """
    system_prompt, user_prompt = EXTRACT_PROMPT(letter_text)
    try:
        llm_response = ask_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,  # Strict extraction, so temperature=0
        )
        response_content = llm_response.choices[0].message.content

        match = re.search(r'```json\n(.*?)```', response_content, re.DOTALL)
        json_string = match.group(1).strip() if match else response_content.strip()

        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON for letter: {letter_text[:50]}...")
        print(f"Error: {e}")
        print(f"Raw LLM response: {response_content}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during extraction for letter: {letter_text[:50]}...")
        print(f"Error: {e}")
        return None


# ---------------------------------------------------------------------------
# 3. DECISION-SUPPORT BRIEF
# ---------------------------------------------------------------------------
# Evolution note: combines the raw letter with the structured extraction output
# to produce a four-part brief (Strengths / Risks / Missing Information /
# Suggested Next Step). The system prompt explicitly forbids "approve"/"reject"
# language, keeping the final lending decision with a human loan officer.

def BRIEF_PROMPT(letter_text, extracted_data):
    """
    Generate a decision-support brief for a loan application, combining the
    original letter with its extracted structured data. Never outputs an
    approve/reject decision — only a suggested next step for a human officer.
    """
    system_prompt = (
        "You are an assistant to a microfinance loan officer. Your task is to provide a decision support brief "
        "for a loan application. Your brief should be factual, neutral, and based solely on the provided letter "
        "and extracted data. Do not make any final approval or rejection decisions; instead, suggest concrete "
        "next steps for the loan officer. Structure your response as follows:\n\n"
        "**Strengths:**\n- [Bullet point 1]\n- [Bullet point 2]\n...\n"
        "**Risks/Red Flags:**\n- [Bullet point 1]\n- [Bullet point 2]\n...\n"
        "**Missing Information:**\n- [Missing piece 1]\n- [Missing piece 2]\n...\n"
        "**Suggested Next Step:** [e.g., 'Invite for interview', 'Request sales records', 'Flag for senior review']"
    )

    user_prompt = (
        f"Please generate a decision support brief for the following loan application letter and extracted data.\n\n"
        f"**Loan Application Letter:**\n{letter_text}\n\n"
        f"**Extracted Data (JSON):**\n{json.dumps(extracted_data, indent=2)}\n\n"
        "Generate the brief following the specified format. Remember, do not make approval or rejection decisions."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    return response.choices[0].message.content
