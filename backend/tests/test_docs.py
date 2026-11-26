import re

from app import create_app


def test_halaman_docs_dan_openapi():
    client = create_app("sqlite://").test_client()
    assert b"swagger-ui" in client.get("/api/docs").data
    r = client.get("/api/openapi.yaml")
    assert r.status_code == 200 and r.data.startswith(b"openapi: 3.0.3")


def test_semua_endpoint_ada_di_openapi():
    app = create_app("sqlite://")
    isi = app.test_client().get("/api/openapi.yaml").get_data(as_text=True)
    for rule in app.url_map.iter_rules():
        if not rule.rule.startswith("/api/") or rule.rule in ("/api/docs", "/api/openapi.yaml"):
            continue
        path = re.sub(r"<(?:\w+:)?(\w+)>", r"{\1}", rule.rule)  # /api/orang/<int:orang_id> -> /api/orang/{orang_id}
        assert f"\n  {path}:" in isi, f"{path} belum didokumentasikan di openapi.yaml"
        for metode in rule.methods - {"HEAD", "OPTIONS"}:
            bagian = isi.split(f"\n  {path}:")[1].split("\n  /api/")[0]
            assert f"\n    {metode.lower()}:" in bagian, f"{metode} {path} belum didokumentasikan"
