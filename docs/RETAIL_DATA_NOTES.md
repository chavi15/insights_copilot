# Online Retail data notes

## Source and licence

UCI Machine Learning Repository, Online Retail dataset (id 352), licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

Citation, as given on the dataset page: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.

The dataset page describes it as all transactions between 01/12/2010 and 09/12/2011 for a UK-based, registered, non-store online retailer that mainly sells unique all-occasion gifts. Many of its customers are wholesalers.

`python -m src.retail_load` downloads `https://archive.ics.uci.edu/static/public/352/online+retail.zip` into `data/raw/` if it is missing, reads `Online Retail.xlsx` with pandas (this needs `openpyxl`), and writes `data/retail.duckdb`. Neither file is committed.

## Row counts

Counts from the file downloaded on 2026-10-04. The loader prints them on every run.

| Step | Count |
| --- | --- |
| Raw rows | 541,909 |
| Exact duplicate rows removed | 5,268 rows |
| Bad-debt adjustment rows removed (invoice starts with A) | 3 rows |
| Zero or negative unit price rows removed | 2,510 rows (1,336 of them with a negative quantity) |
| Cancellation lines kept and flagged | 9,251 rows on 3,836 invoices |
| Lines with a missing customer ID, kept as NULL | 132,564 rows on 1,609 invoices |
| Stock codes upper-cased | 1,974 rows |
| Invoices with more than one timestamp | 43 invoices |
| Customers seen in more than one country | 8 customers |
| **dim_customer** | 4,371 rows |
| **dim_product** | 3,827 rows |
| **invoices** | 23,795 rows |
| **invoice_lines** | 534,128 rows |

541,909 − 5,268 − 3 − 2,510 = 534,128 invoice lines.

## Cleaning decisions

The steps run in this order.

1. **Exact duplicate rows: removed (5,268).** These rows match on every column, including the timestamp to the minute. The file has no line number, so an exact copy cannot be told apart from a genuine second scan of the same item at the same price in the same minute. Treating them as duplicates is the usual choice for this dataset. It lowers totals slightly if some were genuine.
2. **Bad-debt adjustments: removed (3 rows).** Invoices `A563185`, `A563186` and `A563187` carry stock code `B`, the description "Adjust bad debt" and, in two cases, a unit price of −11,062.06. They are accounting entries, not sales.
3. **Zero or negative unit price: removed (2,510 rows).** Their descriptions are mostly stock notes such as "check", "damaged", "found", "?" or blank. 1,336 of them have a negative quantity and no customer: these are stock write-offs. Another 1,174 have a positive quantity and a zero price, and only 40 of those have a customer ID. Removing them means `unit_price` is always above zero. It also lowers unit totals slightly, because free items are no longer counted.
4. **Cancellations: kept and flagged (9,251 lines, 3,836 invoices).** An invoice number starting with `C` is a cancellation (dataset page: "If this code starts with letter 'c', it indicates a cancellation"). These invoices get `is_cancelled = true`. Their quantity and `line_total` stay negative, so `SUM(line_total)` gives net revenue. A cancellation is a separate invoice, and the dataset does not link it to the original order.
5. **Negative quantities.** After steps 2 and 3, every negative quantity is on a cancellation and every cancellation line is negative. The loader checks both and stops with an error if either is broken.
6. **Missing customer IDs: kept with `customer_id` NULL (132,564 lines, 1,609 invoices).** These are about 25% of lines. Removing them would understate revenue. Customer-level questions leave them out because `dim_customer` only holds known customers.
7. **Stock code case: upper-cased (1,974 rows).** Among the kept lines, 110 lower-case codes have an upper-case twin, such as `85123a` and `85123A`, with the same description, so they are one product. Codes are also trimmed of whitespace.
8. **Product description: most frequent per stock code.** Descriptions are trimmed first (112,312 had extra whitespace). Ties go to the alphabetically first description. Only lines kept after cleaning are used, so write-off notes such as "damaged" do not become product names. No kept product is left without a description.
9. **Invoice date: earliest timestamp.** 43 invoices have lines with more than one timestamp, usually a minute apart.
10. **Customer country: most frequent.** 8 customers appear with more than one country. `dim_customer.country` is their most frequent invoice country, with ties going to the alphabetically first. `invoices.country` keeps each invoice's own country, which is consistent within every invoice.
11. **Not changed.** Non-merchandise stock codes such as POST, DOT, C2 (postage), M (manual), BANK CHARGES, AMAZONFEE and CRUK are kept and count towards revenue. Country names stay as recorded (EIRE, RSA, Unspecified, European Community). 4 lines have a unit price of 0.001 and are kept as they are. The same stock code can appear on several lines of one invoice (5,390 extra lines), so `invoice_lines` has no unique key.

## Known quirks

- The two largest orders in the file, 80,995 units of stock code 23843 and 74,215 units of 23166, were cancelled through separate C invoices. Questions about "units sold on non-cancelled invoices" still count those orders. Net units, which subtract cancellations, do not.
- December 2011 covers only 1 to 9 December, so monthly totals for that month are partial.
- `line_total` is `quantity × unit_price` rounded to 2 decimals, per line.
