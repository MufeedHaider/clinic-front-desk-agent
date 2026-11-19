from conftest import DOCS_DIR


def test_ingest_loads_all_files(kb):
    assert kb.count() > 4


def test_ingest_is_idempotent(kb):
    before = kb.count()
    kb.ingest(DOCS_DIR)  # running the pipeline again must not add duplicates
    assert kb.count() == before


def test_retrieval_finds_the_right_document(kb):
    top_doc, _ = kb.search("What time does the clinic open on Saturday?", k=1)[0]
    assert top_doc.metadata["source"] == "01_about_and_hours.md"

    top_doc, _ = kb.search("How much is the cancellation charge if I reschedule?", k=1)[0]
    assert top_doc.metadata["source"] == "03_appointment_policy.md"
