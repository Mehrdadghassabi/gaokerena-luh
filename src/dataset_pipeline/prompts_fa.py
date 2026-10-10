# Persian (Farsi) prompts for claim extraction / matching / fact-checking.
#
# Mirrors the structure of benchmark_multilingual/{en,zh}/prompts.py.
#
# Two deliberate differences from the English defaults:
#
# 1. Claims must stay in Persian. With the English prompts, ~44% of extracted
#    claims came back translated into English and ~12% mixed, which (a) is
#    inconsistent labelling and (b) makes the downstream matching step a
#    cross-lingual task, since it has to map an English claim back onto Persian
#    words in the original sentence.
#
# 2. Claims are coarser. The English example intentionally emits nested,
#    overlapping claims (3 claims from one sentence, two of them near
#    duplicates), which inflates claim count -- and therefore API calls -- for
#    no extra supervision signal. The Persian example below shows disjoint
#    claims instead, and the instructions forbid restating the same fact.

CLAIM_EXTRACTION_PROMPTS = {
    "fa": """جمله زیر را به ادعاهای مستقل و قابل راستی‌آزمایی تجزیه کن.

قواعد:
- تمام ادعاها باید به زبان فارسی نوشته شوند. هرگز ترجمه نکن.
- هر ادعا باید یک واقعیت مستقل و مجزا باشد.
- ادعاهای تکراری یا تودرتو تولید نکن. یک واقعیت را دو بار به شکل‌های مختلف بازنویسی نکن.
- فقط واقعیت‌هایی را استخراج کن که در خود جمله آمده‌اند. چیزی به آن اضافه نکن.
- اگر جمله یک عنوان، سرتیتر، برچسب یا قطعه ناقص است و هیچ واقعیت قابل راستی‌آزمایی ندارد، خروجی را خالی بگذار.
- هر ادعا را در یک خط جداگانه و با «- » شروع کن.

مثال:
جمله: «او در لندن به دنیا آمد و تا ۱۱ سالگی توسط مادر و پدرش بزرگ شد.»
ادعاها:
- او در لندن به دنیا آمد.
- او تا ۱۱ سالگی توسط مادر و پدرش بزرگ شد.

مثال:
جمله: «خواص پوستی:»
ادعاها:

جمله: «{sent}»
ادعاها:""",
}


MATCHING_PROMPTS = {
    "fa": (
        "با توجه به واقعیت داده‌شده، کلمات متناظر در جمله اصلی را که به "
        "استنتاج این واقعیت کمک می‌کنند مشخص کن. تمام کلمات مرتبط با واقعیت "
        "را به همان ترتیبی که در جمله اصلی ظاهر شده‌اند فهرست کن و هر کلمه "
        "را با کاما جدا کن.\n"
        "کلمات را دقیقاً به همان شکلی که در جمله آمده‌اند بنویس؛ شکل کلمه را "
        "تغییر نده، ترجمه نکن و کلمه‌ای که در جمله نیست اضافه نکن.\n"
        "فقط فهرست کلمات جدا شده با کاما را بنویس و هیچ توضیح اضافه‌ای نده.\n"
        "واقعیت: {claim}\n"
        "جمله: {sent}\n"
        "کلمات جمله که به استنتاج واقعیت کمک می‌کنند، جدا شده با کاما: "
    )
}


OPENAI_FACT_CHECK_PROMPTS = {
    "fa": (
        """سؤال: {input}

با توجه به معتبرترین و به‌روزترین منابع اطلاعاتی، مشخص کن که آیا تمام اطلاعات ارائه‌شده در ادعای زیر درست است یا نه.

ادعا: {claim}
"""
    ),
}


# NB: the repo's get_prompts() reads this as `mod.FACT_CHECK_SUMMARIZE_PROMPT`
# while lm_polygraph exports it as OPENAI_FACT_CHECK_SUMMARIZE_PROMPT, so both
# names are bound here.
OPENAI_FACT_CHECK_SUMMARIZE_PROMPT = {
    "fa": (
        """سؤال: {input}

ادعا: {claim}

آیا ادعای بالا درست است؟

پاسخ: {reply}

این پاسخ را در یک کلمه خلاصه کن و بگو آیا ادعا درست است: "True"، "False" یا "Not known".
"""
    ),
}

FACT_CHECK_SUMMARIZE_PROMPT = OPENAI_FACT_CHECK_SUMMARIZE_PROMPT
