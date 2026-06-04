from src.paper import ResearchPaper


def main() -> None:
    

    paper = ResearchPaper(
        title = "Test Research Paper",
        authors = ["Sameer", "Ananya"],
        abstract = "Test Abstract",
        keywords = ["Test", "Wassup", "Kick"],
        section_headings = ["Section 1", "SEction 2"],
        page_count = 0,
        word_count = 0,
        full_text = "Full Text",
        references = ["Grishmita", "Dhanvi"],
        doi = "001100",
        publication_year=2026,
        venue="what",
        source_path="/hello"


    )

    print(paper)


if __name__ == "__main__":
    main()
