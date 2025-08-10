import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

async def main():
    urls = [
        "https://wiki.student.uni-goettingen.de/support/drucken/druckpreise",
    ]

    # Bilder im Markdown generell unterdrücken
    md_gen = DefaultMarkdownGenerator(options={"ignore_images": True})

    run_cfg = CrawlerRunConfig(
        excluded_tags=["header", "footer"],          # Kopf- und Fußzeile entfernen
        excluded_selector=(
            "nav#dokuwiki__aside, "                 # linke Seitenleiste
            "nav#dokuwiki__pagetools, "             # rechte Werkzeugleiste
            "a.media, img, "                        # Bildlinks + <img>-Tags
            "div#dw__toc"                           # TOC-Block
        ),
        markdown_generator=md_gen,
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun_many(urls, config=run_cfg)

        for url, res in zip(urls, results):
            print(f"\n--- {url} ---\n")
            print(res.markdown[:600])               # erste 600 Zeichen ohne Header, Footer, TOC & Bilder

if __name__ == "__main__":
    asyncio.run(main())
