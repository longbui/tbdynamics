from pathlib import Path
import pylatex as pl
from pylatex.section import Section, Subsection
from pylatex.utils import NoEscape
import matplotlib.figure as mpl
import plotly.graph_objects as go
import pandas as pd

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"


def escape_special_text(
    text: str,
) -> str:
    """
    Don't escape characters if they are needed for citations or underscores.

    Args:
        text: Text for document

    Returns:
        Revised text string
    """
    return NoEscape(text) if "\cite{" in text or "extunderscore" in text or "$" in text else text


class DocElement:
    """
    Abstract class for creating a model with accompanying TeX documentation.
    """
    def __init__(self):
        pass

    def emit_latex(
            self, 
            doc: pl.document.Document,
        ):
        """
        Output contents of documentation object to working document.

        Args:
            doc: Working TeX document
        """
        pass


class TextElement(DocElement):
    """
    Paragraph text documentation object with method for TeX output.

    Args:
        DocElement: Any documentation object    
    """
    def __init__(
            self, 
            text: str,
        ):
        """
        Set up object with text input.

        Args:
            text: The text to write
        """
        self.text = escape_special_text(text)

    def emit_latex(
            self, 
            doc: pl.document.Document,
        ):
        """
        Write the text to the document.

        Args:
            doc: The PyLaTeX object to add to
        """
        doc.append(self.text)


class FigElement(DocElement):
    """
    Figure documentation object with method for TeX output.

    Args:
        DocElement: Any documentation object    
    """
    def __init__(
            self, 
            fig_name: str,
            caption: str="",
            resolution: str="350px",
        ):
        """
        Set up object with figure input and other requests.

        Args:
            fig_name: The label of the figure to write
            caption: Figure caption
            resolution: Resolution to write to
        """
        self.fig_name = fig_name
        self.caption = caption
        self.resolution = resolution
    
    def emit_latex(
            self, 
            doc: pl.document.Document,
        ):
        with doc.create(pl.Figure()) as plot:
            plot.add_image(self.fig_name, width=self.resolution)
            plot.add_caption(self.caption)


class TableElement(DocElement):
    """
    Table documentation object with method for TeX output.

    Args:
        DocElement: Any documentation object
    """
    
    def __init__(
        self, 
        input_table: pd.DataFrame,
        table_width: float=11.0,
        col_requests: list=[], 
    ):
        """
        Set up object with figure input and other requests.

        Args:
            input_table: Table that needs to go into document
            table_width: Total width of the table on the page
            col_widths: Requested proportional widths of the columns within the table
        """
        if col_requests:
            width_req_strs = [str(round(col_w * table_width, 2)) for col_w in col_requests]
        else:
            n_cols = input_table.shape[1] + 1
            width_req_strs = [str(round(table_width / n_cols, 2))] * n_cols
        self.col_widths = "p{" + "cm} p{".join(width_req_strs) + "cm}"
        self.table = input_table

    def emit_latex(
            self, 
            doc: pl.document.Document,
        ):
        with doc.create(pl.LongTable(self.col_widths)) as output_table:
            headers = [""] + list(self.table.columns)
            output_table.add_row(headers)
            output_table.add_hline()
            for index in self.table.index:
                content = [index] + [escape_special_text(str(element)) for element in self.table.loc[index]]
                output_table.add_row(content)
                output_table.add_hline()
            doc.append(pl.NewPage())


def add_element_to_document(
    section_name: str, 
    element: DocElement, 
    doc_sections: dict,
    subsection_name: str="no_subsection",
):
    """
    Add a document element to the working document compilation structure.

    Args:
        section_name: Title of the document section to add the element to
        element: The element to add
        doc_sections: The document to be added to
        subsection_name: The title of the sub-section to add to the element to, if any
    """
    if section_name not in doc_sections:
        doc_sections[section_name] = {}
    if subsection_name not in doc_sections[section_name]:
        doc_sections[section_name][subsection_name] = []
    doc_sections[section_name][subsection_name].append(element)


def save_pyplot_add_to_doc(
    plot: mpl.Figure, 
    plot_name: str, 
    section_name: str, 
    working_doc: dict, 
    caption: str="", 
    dpi: float=250,
):
    """
    Save a matplotlib figure to a standard location and add it to the working document.
    
    Args:
        plot: The figure object
        plot_name: Name to assign the file
        section_name: Section to add the figure to
        working_doc: Working document
        caption: Optional caption to add
        dpi: Resolution to save the figure at
    """
    plot.savefig(SUPPLEMENT_PATH / f"{plot_name}.jpg", dpi=dpi)
    add_element_to_document(section_name, FigElement(plot_name, caption=caption), working_doc)


def save_plotly_add_to_doc(
    plot: go.Figure, 
    plot_name: str, 
    section_name: str, 
    working_doc: dict, 
    caption="", 
    scale=4.0,
):
    """
    Save a plotly figure to a standard location and add it to the working document.

    Args:
        plot: The figure object
        plot_name: Name to assign the file
        section_name: Section to add the figure to
        working_doc: Working document
        caption: Optional caption to add
        dpi: Resolution adjuster for saving the figure
    """
    plot.write_image(SUPPLEMENT_PATH / f"{plot_name}.jpg", scale=scale)
    add_element_to_document(section_name, FigElement(plot_name, caption=caption), working_doc)


def generate_doc(
    title: str, 
    bib_filename: str,
) -> pl.document.Document:
    """
    Use PyLaTeX to prepare a TeX file representing a document for filling with content later.

    Args:
        title: The title to go into the document
        bib_filename: The filename for the .bib BibTeX file to get the references from

    Returns:
        The document object
    """
    doc = pl.Document()
    doc.preamble.append(pl.Package("biblatex", options=["sorting=none"]))
    doc.preamble.append(pl.Package("booktabs"))
    doc.preamble.append(pl.Command("addbibresource", arguments=[f"{bib_filename}.bib"]))
    doc.preamble.append(pl.Command("title", title))
    doc.append(NoEscape(r"\maketitle"))
    return doc


def compile_doc(
    doc_sections: dict, 
    doc: pl.document.Document,
):
    """
    Compile the full PyLaTeX document from the dictionary
    of elements by section requested.

    Args:
        doc_sections: Working compilation of instruction elements
        doc: The TeX file to create from the elements
    """
    for section in doc_sections:
        with doc.create(Section(section)):
            if "no_subsection" in doc_sections[section]:
                for element in doc_sections[section]["no_subsection"]:
                    element.emit_latex(doc)
            for subsection in [sub for sub in doc_sections[section].keys() if sub != "no_subsection"]:
                with doc.create(Subsection(subsection)):
                    for element in doc_sections[section][subsection]:
                        element.emit_latex(doc)
            doc.append(pl.NewPage())
    doc.append(pl.Command("printbibliography"))
    doc.generate_tex(str(SUPPLEMENT_PATH / "supplement"))
