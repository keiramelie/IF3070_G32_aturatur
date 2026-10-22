from __future__ import annotations

import csv
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.core.models import Ship, Vehicle

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Header aliases are normalized first: spaces, underscores, punctuation, and
# capitalization do not matter. Add aliases here if the official dataset uses
# different names; the optimization algorithms do not need to change.
ALIASES = {
    "vehicle_id": {
        "vehicleid", "id", "idvehicle", "idkendaraan", "kodekendaraan",
        "vehicle", "vehicleno", "carid", "truckid", "nomorkendaraan",
    },
    "width": {
        "width", "vehiclewidth", "carwidth", "truckwidth", "lebar",
        "lebarkendaraan", "lebarkapal", "shipwidth", "deckwidth", "vesselwidth",
    },
    "length": {
        "length", "vehiclelength", "carlength", "trucklength", "panjang",
        "panjangkendaraan", "panjangkapal", "shiplength", "decklength", "vessellength",
    },
    "shipping_fee": {
        "shippingfee", "freightfee", "freightcost", "shippingcost", "fee",
        "revenue", "profit", "price", "ongkir", "biayapengiriman", "tarifpengiriman",
        "pendapatanpengiriman",
    },
    "weight": {
        "weight", "weightkg", "vehicleweight", "mass", "masskg", "berat",
        "beratkendaraan", "beban",
    },
    "eta": {
        "eta", "estimatedtime", "estimatedtimeofarrival", "arrivalestimate",
        "waktukedatangan", "estimasiwaktu", "harikedatangan",
    },
    "max_capacity": {
        "maxcapacity", "capacity", "weightcapacity", "maxweight", "maximumweight",
        "shipcapacity", "shipmaxcapacity", "kapasitas", "kapasitasmaksimum",
        "kapasitasberat", "dayaangkut",
    },
}


def _normalize_header(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        sample = file.read(4096)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(file, dialect=dialect)
        if not reader.fieldnames:
            return []
        return [dict(row) for row in reader if any(str(v or "").strip() for v in row.values())]


def _read_excel(path: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            f"File {path.name} adalah Excel. Untuk membacanya, install dependensi: "
            "py -3.14 -m pip install pandas openpyxl"
        ) from exc

    result = []
    sheets = pd.read_excel(path, sheet_name=None)
    for sheet_name, frame in sheets.items():
        frame = frame.dropna(how="all")
        if not frame.empty:
            result.append((f"{path.name}:{sheet_name}", frame.fillna("").to_dict(orient="records")))
    return result


def _read_tables(dataset_dir: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")

    tables: list[tuple[str, list[dict[str, Any]]]] = []
    for path in sorted(dataset_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.suffix.lower() == ".csv":
            rows = _read_csv(path)
            if rows:
                tables.append((path.name, rows))
        else:
            tables.extend(_read_excel(path))

    if not tables:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"Tidak ada tabel dataset yang bisa dibaca di {dataset_dir}. "
            f"Letakkan file {extensions} di folder tersebut."
        )
    return tables


def _column_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    if not rows:
        return {}
    actual_headers = list(rows[0].keys())
    normalized_to_actual = {_normalize_header(header): header for header in actual_headers}
    found = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized_to_actual:
                found[canonical] = normalized_to_actual[alias]
                break
    return found


def _is_ship_table(rows: list[dict[str, Any]]) -> bool:
    columns = _column_map(rows)
    return all(key in columns for key in ("width", "length", "max_capacity")) and len(rows) == 1


def _is_vehicle_table(rows: list[dict[str, Any]]) -> bool:
    columns = _column_map(rows)
    required = ("vehicle_id", "width", "length", "shipping_fee", "weight")
    return bool(rows) and all(key in columns for key in required)


def _value(row: dict[str, Any], column: str, label: str, row_number: int) -> Any:
    raw = row.get(column)
    if raw is None or str(raw).strip() == "":
        raise ValueError(f"Nilai '{label}' kosong pada baris data {row_number}.")
    return raw


def _number(raw: Any, label: str, row_number: int) -> float:
    text = str(raw).strip().replace(" ", "")
    # Support decimal comma in semicolon-delimited files, e.g. 12,5.
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Nilai '{label}' pada baris data {row_number} harus berupa angka; "
            f"nilai ditemukan: {raw!r}."
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"Nilai '{label}' pada baris data {row_number} harus angka terbatas.")
    return number


def _integer(raw: Any, label: str, row_number: int) -> int:
    number = _number(raw, label, row_number)
    if not number.is_integer():
        raise ValueError(
            f"'{label}' pada baris data {row_number} harus bilangan bulat, "
            f"karena grid penempatan memakai satuan integer; ditemukan {raw!r}."
        )
    return int(number)


def load_problem(dataset_dir: Path | str) -> tuple[Ship, dict[str, Vehicle]]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    tables = _read_tables(dataset_dir)

    ship_candidates = [(name, rows) for name, rows in tables if _is_ship_table(rows)]
    vehicle_candidates = [(name, rows) for name, rows in tables if _is_vehicle_table(rows)]

    if len(ship_candidates) != 1:
        available = "; ".join(f"{name}: {list(rows[0].keys()) if rows else []}" for name, rows in tables)
        raise ValueError(
            "Harus ditemukan tepat satu tabel kapal dengan kolom lebar, panjang, dan kapasitas. "
            f"Ditemukan {len(ship_candidates)}. Tabel terdeteksi: {available}"
        )
    if len(vehicle_candidates) != 1:
        available = "; ".join(f"{name}: {list(rows[0].keys()) if rows else []}" for name, rows in tables)
        raise ValueError(
            "Harus ditemukan tepat satu tabel kendaraan dengan kolom ID, lebar, panjang, "
            "shipping fee, dan berat. ETA opsional. "
            f"Ditemukan {len(vehicle_candidates)}. Tabel terdeteksi: {available}"
        )

    ship_name, ship_rows = ship_candidates[0]
    ship_row = ship_rows[0]
    ship_columns = _column_map(ship_rows)
    ship = Ship(
        width=_integer(_value(ship_row, ship_columns["width"], "width kapal", 2), "width kapal", 2),
        length=_integer(_value(ship_row, ship_columns["length"], "length kapal", 2), "length kapal", 2),
        max_capacity=_number(
            _value(ship_row, ship_columns["max_capacity"], "kapasitas kapal", 2),
            "kapasitas kapal", 2,
        ),
    )

    vehicle_name, vehicle_rows = vehicle_candidates[0]
    vehicle_columns = _column_map(vehicle_rows)
    vehicles: dict[str, Vehicle] = {}
    for row_number, row in enumerate(vehicle_rows, start=2):
        vehicle_id = str(_value(row, vehicle_columns["vehicle_id"], "ID kendaraan", row_number)).strip()
        if vehicle_id in vehicles:
            raise ValueError(f"ID kendaraan duplikat {vehicle_id!r} pada {vehicle_name}, baris {row_number}.")

        eta = 0
        if "eta" in vehicle_columns and str(row.get(vehicle_columns["eta"], "")).strip():
            eta = _integer(row[vehicle_columns["eta"]], "ETA", row_number)

        vehicle = Vehicle(
            id=vehicle_id,
            width=_integer(_value(row, vehicle_columns["width"], "width kendaraan", row_number), "width kendaraan", row_number),
            length=_integer(_value(row, vehicle_columns["length"], "length kendaraan", row_number), "length kendaraan", row_number),
            shipping_fee=_number(_value(row, vehicle_columns["shipping_fee"], "shipping fee", row_number), "shipping fee", row_number),
            weight=_number(_value(row, vehicle_columns["weight"], "berat kendaraan", row_number), "berat kendaraan", row_number),
            eta=eta,
        )
        if vehicle.width <= 0 or vehicle.length <= 0:
            raise ValueError(f"Dimensi kendaraan harus positif: {vehicle_id}")
        if vehicle.weight < 0 or vehicle.shipping_fee < 0 or vehicle.eta < 0:
            raise ValueError(f"Berat, shipping fee, dan ETA tidak boleh negatif: {vehicle_id}")
        vehicles[vehicle_id] = vehicle

    if ship.width <= 0 or ship.length <= 0 or ship.max_capacity < 0:
        raise ValueError("Dimensi kapal harus positif dan kapasitas tidak boleh negatif.")
    if not vehicles:
        raise ValueError("Tabel kendaraan tidak berisi data.")

    print(f"Dataset kapal terbaca dari: {ship_name}")
    print(f"Dataset kendaraan terbaca dari: {vehicle_name}")
    return ship, vehicles
