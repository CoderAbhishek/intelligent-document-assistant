"""A synthetic 10-page annual report used only for evaluation.

Meridian Harbor Logistics is fictional and every fact here is invented. Look-alike facts
(two terminals with the same kinds of attributes) are spread over different pages so
retrieval has to tell similar passages apart. Results on this document say nothing about
performance on real-world PDFs.
"""

from pathlib import Path

import pymupdf

PAGES = [
    # 1
    "Meridian Harbor Logistics - Annual Report 2024. Meridian Harbor Logistics is a port "
    "operator and freight forwarder headquartered in the city of Valmora. The company handles "
    "container cargo, bulk grain and temperature-controlled goods through two terminals and a "
    "network of inland depots. In 2024 the company handled 1.84 million twenty-foot equivalent "
    "units (TEU) across its terminals and employed 3,120 people. This report covers the "
    "financial year ending 31 December 2024. Statements about the future are expectations, "
    "not guarantees.",
    # 2
    "Meridian Harbor Logistics was founded in 1987 by the brothers Anselm and Tobias Rourke, "
    "who began with a single leased berth and two cranes. The North Quay terminal opened in "
    "1994 and remained the company's only terminal for nearly three decades. In 2009 the "
    "company was listed on the Valmora Stock Exchange. The South Basin terminal, built to "
    "serve larger vessels, began operations in 2022 after a four-year construction programme. "
    "Tobias Rourke retired from the board in 2016; Anselm Rourke served as chairman until 2021.",
    # 3
    "The chief executive officer is Priya Nandakumar, who joined the company in 2018 and became "
    "CEO in 2021. The chief financial officer is Lars Eklund, appointed in 2020. The chair of "
    "the board is Marguerite Oyelaran, who succeeded Anselm Rourke in 2021. The board has nine "
    "members, of whom six are independent. The audit committee is chaired by Dev Prakash. "
    "Executive pay is reviewed annually by the remuneration committee, and the long-term "
    "incentive plan is linked to safety and emissions targets as well as profit.",
    # 4
    "North Quay terminal has four berths and a quay length of 1,150 metres. It has an annual "
    "design capacity of 1.1 million TEU and handled 0.96 million TEU in 2024. North Quay "
    "receives container ships of up to 8,000 TEU. The terminal operates 11 ship-to-shore "
    "cranes, the oldest of which was installed in 1998. It is connected to the national rail "
    "network by a double-track line, and about 28 per cent of containers leave the terminal "
    "by rail.",
    # 5
    "South Basin terminal has three berths and a quay length of 980 metres. It has an annual "
    "design capacity of 1.4 million TEU and handled 0.88 million TEU in 2024, its second full "
    "year of operation. South Basin can receive ships of up to 18,000 TEU thanks to a dredged "
    "channel with a depth of 16.5 metres. The terminal uses 9 automated stacking cranes and 8 "
    "ship-to-shore cranes. Rail connections are limited: about 9 per cent of containers leave "
    "by rail, and the remainder by road.",
    # 6
    "Group revenue for 2024 was 612 million euros, an increase of 7 per cent on 2023. Port "
    "operations generated 438 million euros and the freight forwarding division generated 174 "
    "million euros. Operating profit was 91 million euros, giving an operating margin of 14.9 "
    "per cent. Net profit after tax was 63 million euros. The board proposed a dividend of "
    "0.42 euros per share. Net debt at year end was 305 million euros, down from 340 million "
    "euros a year earlier.",
    # 7
    "The freight forwarding division operates 14 inland depots and a fleet of 260 trucks. Its "
    "cold chain service, launched in 2019, uses 85 refrigerated trailers and stores "
    "temperature-controlled goods at the Harrowgate depot, which has 12,000 square metres of "
    "chilled storage. In 2024 cold chain volumes grew by 18 per cent. Most refrigerated cargo "
    "consists of fruit and dairy products. The division delivers to customers in six "
    "countries.",
    # 8
    "Safety is the company's first priority. The lost-time injury frequency rate (LTIFR) was "
    "1.9 per million hours worked in 2024, compared with 2.6 in 2023. The target for 2025 is "
    "1.5. There were no fatalities in 2024. Following two crane-related incidents at North "
    "Quay in 2023, all crane operators completed additional training in 2024. The company "
    "runs a confidential reporting line that is available in four languages.",
    # 9
    "The company aims to cut its direct carbon emissions by 40 per cent by 2030, measured "
    "against 2019 levels. In 2024 emissions were 22 per cent below the 2019 level. Shore power "
    "for ships at berth is available at South Basin but not yet at North Quay, where "
    "installation is planned for 2026. The company is replacing diesel yard vehicles with "
    "electric ones; 35 of 120 vehicles had been replaced by the end of 2024. About 61 per cent "
    "of the electricity used came from renewable sources.",
    # 10
    "The main risks identified by the board are a slowdown in global trade, rising interest "
    "rates on the company's floating-rate debt, labour disputes at ports, and cyber attacks on "
    "terminal systems. For 2025 the company expects South Basin volumes to grow faster than "
    "North Quay volumes as shipping lines move to larger vessels. Capital expenditure for 2025 "
    "is planned at 120 million euros. The company does not publish revenue guidance for 2025.",
]

PDF_NAME = "meridian_harbor_annual_report_2024.pdf"


def build_pdf(directory: Path) -> Path:
    """Write the corpus as a PDF (one page per entry) into ``directory``."""
    path = directory / PDF_NAME
    with pymupdf.open() as doc:
        for text in PAGES:
            page = doc.new_page()
            if page.insert_textbox(pymupdf.Rect(72, 72, 523, 770), text) < 0:
                raise ValueError("Corpus page does not fit on one PDF page")
        doc.save(path)
    return path
