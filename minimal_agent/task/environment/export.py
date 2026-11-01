"""Nightly export: prints the orders as CSV."""
import csv
import json
import sys
from datetime import date
from pathlib import Path

ORDERS = Path(__file__).parent / "orders.json"


def main():
    orders = json.loads(ORDERS.read_text())
    writer = csv.writer(sys.stdout)
    writer.writerow(["id", "customer", "created", "shipped", "total"])
    for order in orders:
        created = date.fromisoformat(order["created"])
        shipped = date.fromisoformat(order["shipped"])
        writer.writerow([order["id"], order["customer"], created.strftime("%m/%d/%Y"),
                         shipped.strftime("%m/%d/%Y"), f"{order['total']:.2f}"])


if __name__ == "__main__":
    main()
