from pathlib import Path
import pandas as pd
import yaml as yml
from abc import abstractmethod, ABC


def get_tex_formatted_date(date):
    date_of_month = date.strftime('%d').lstrip('0')
    special_cases = {
        '1': 'st', 
        '2': 'nd',
        '3': 'rd',
        '21': 'st',
        '22': 'nd',
        '23': 'rd',
        '31': 'st',
    }
    text_super = special_cases[date_of_month] if date_of_month in special_cases else 'th'
    return f'{date_of_month}\\textsuperscript{{{text_super}}}{date.strftime(" of %B %Y")}'


class TexDoc(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def add_line(self, line: str, section: str, subsection: str=''):
        pass

    @abstractmethod
    def prepare_doc(self):
        pass

    @abstractmethod
    def write_doc(self, order: list=[]):
        pass

    @abstractmethod
    def emit_doc(self, section_order: list=[]) -> str:
        pass

    @abstractmethod
    def include_figure(self, caption: str, filename: str, section: str, subsection: str=''):
        pass

    @abstractmethod
    def include_table(self, table: pd.DataFrame, section: str, subsection: str='', col_splits=None, table_width=14.0, longtable=False):
        pass

    @abstractmethod
    def save_content(self):
        pass

    @abstractmethod
    def load_content(self):
        pass


class DummyTexDoc(TexDoc):
    def add_line(self, line: str, section: str, subsection: str=''):
        pass

    def prepare_doc(self):
        pass

    def write_doc(self, order: list=[]):
        pass

    def emit_doc(self, section_order: list=[]) -> str:
        pass

    def include_figure(self, caption: str, filename: str, section: str, subsection: str=''):
        pass

    def include_table(self, table: pd.DataFrame, section: str, subsection: str='', col_splits=None, table_width=14.0, longtable=False):
        pass

    def save_content(self):
        pass

    def load_content(self):
        pass


class ConcreteTexDoc:
    def __init__(
        self, 
        path: Path, 
        doc_name: str, 
        title: str, 
        bib_filename: str,
    ):
        """
        Object that can do the basic collation and emitting of a TeX-formatted
        string, including basic features for figures and tables.

        Args:
            path: Path for writing the output document
            doc_name: Filename for the document produced
            title: Title to go in the document
            bib_filename: Name of the bibliography file
        """
        self.content = {}
        self.path = path
        self.doc_name = doc_name
        self.bib_filename = bib_filename
        self.title = title
        self.prepared = False
        self.standard_sections = ['preamble', 'endings']

    def add_line(
        self, 
        line: str, 
        section: str, 
        subsection: str='',
    ):
        """
        Add a single line string to the appropriate section and subsection 
        of the working document.

        Args:
            line: The TeX line to write
            section: The heading of the section for the line to go into
            subsection: The heading of the subsection for the line to go into
        """
        if section not in self.content:
            self.content[section] = {}
        if not subsection:
            if '' not in self.content[section]:
                self.content[section][''] = []
            self.content[section][''].append(line)
        else:
            if subsection not in self.content[section]:
                self.content[section][subsection] = []
            self.content[section][subsection].append(line)
        
    def prepare_doc(self):
        """
        Essentially blank method for overwriting in parent class.
        """
        self.prepared = True

    def write_doc(self, order: list=[]):
        """
        Write the compiled document string to disc.
        """
        with open(self.path / f'{self.doc_name}.tex', 'w') as doc_file:
            doc_file.write(self.emit_doc(section_order=order))
    
    def emit_doc(
        self, 
        section_order: list=[],
    ) -> str:
        """
        Collate all the sections together into the big string to be outputted.

        Arguments:
            section_order: The order to write the document sections in
        Returns:
            The final text to write into the document
        """
        content_sections = sorted([s for s in self.content if s not in self.standard_sections])
        if section_order and sorted(section_order) != content_sections:
            msg = 'Sections requested are not those in the current contents'
            raise ValueError(msg)

        order = section_order if section_order else self.content.keys()

        if not self.prepared:
            self.prepare_doc()
        final_text = ''
        for line in self.content['preamble']['']:
            final_text += f'{line}\n'
        for section in [k for k in order if k not in self.standard_sections]:
            final_text += f'\n\\section{{{section}}} \\label{{{section.lower().replace(" ", "_")}}}\n'
            if '' in self.content[section]:
                for line in self.content[section]['']:
                    final_text += f'{line}\n'
            for subsection in [k for k in self.content[section].keys() if k != '']:
                final_text += f'\n\\subsection{{{subsection}}} \\label{{{subsection.lower().replace(" ", "_")}}}\n'
                for line in self.content[section][subsection]:
                    final_text += f'{line}\n'
        for line in self.content['endings']['']:
            final_text += f'{line}\n'
        return final_text

    def include_figure(
        self, 
        caption: str, 
        filename: str, 
        section: str, 
        subsection: str='',
    ):
        """
        Add a figure with standard formatting to the document.

        Args:
            caption: Figure caption
            filename: Filename for finding the image file
            section: The heading of the section for the figure to go into
            subsection: The heading of the subsection for the figure to go into
        """
        self.add_line('\\begin{figure}', section, subsection)
        self.add_line(f'\\caption{{{caption}}}', section, subsection)
        self.add_line(f'\\includegraphics[width=\\textwidth]{{{filename}.jpg}}', section, subsection)
        self.add_line(f'\\label{{{filename}}}', section, subsection)
        self.add_line('\\end{figure}', section, subsection)

    def include_table(
        self, 
        table: pd.DataFrame, 
        section: str, 
        subsection: str='', 
        col_splits=None, 
        table_width=14.0, 
        longtable=False,
    ):
        """
        Use a dataframe to add a table to the working document.

        Args:
            table: The table to be written
            section: The heading of the section for the figure to go into
            subsection: The heading of the subsection for the figure to go into
            widths: Optional user request for columns widths if not evenly distributed
            table_width: Overall table width if widths not requested
            longtable: Whether to use the longtable module to span pages
        """
        n_cols = table.shape[1] + 1
        if not col_splits:
            splits = [round(1.0 / n_cols, 4)] * n_cols
        elif len(col_splits) != n_cols:
            raise ValueError('Wrong number of proportion column splits requested')
        else:
            splits = col_splits
        col_widths = [w * table_width for w in splits]
        col_format_str = ' '.join([f'>{{\\raggedright\\arraybackslash}}p{{{width}cm}}' for width in col_widths])
        table_text = table.style.to_latex(
            column_format=col_format_str,
            hrules=True,
        )
        table_text = table_text.replace('{tabular}', '{longtable}') if longtable else table_text
        self.add_line('\\begin{center}', section, subsection=subsection)
        self.add_line(table_text, section, subsection=subsection)
        self.add_line('\end{center}', section, subsection=subsection)

    def save_content(self):
        with open(self.path / f'{self.doc_name}.yml', 'w') as file:
            yml.dump(self.content, file)

    def load_content(self):
        with open(self.path / f'{self.doc_name}.yml', 'r') as file:
            self.content = yml.load(file, Loader=yml.FullLoader)


class StandardTexDoc(ConcreteTexDoc):
    def prepare_doc(self):
        """
        Add packages and text that standard documents need to include the other features.
        """
        self.prepared = True
        self.add_line('\\documentclass{article}', 'preamble')

        # Packages that don't require arguments
        standard_packages = [
            'hyperref',
            'biblatex',
            'graphicx',
            'longtable',
            'booktabs',
            'array',
        ]
        for package in standard_packages:
            self.add_line(f'\\usepackage{{{package}}}', 'preamble')

        self.add_line(r'\usepackage[a4paper, total={15cm, 20cm}]{geometry}', 'preamble')
        self.add_line(r'\usepackage[labelfont=bf,it]{caption}', 'preamble')
        self.add_line(f'\\addbibresource{{{self.bib_filename}.bib}}', 'preamble')
        self.add_line(f'\\title{{{self.title}}}', 'preamble')
        self.add_line('\\begin{document}', 'preamble')
        self.add_line('\date{}', 'preamble')
        self.add_line('\maketitle', 'preamble')
        
        self.add_line('\\printbibliography', 'endings')
        self.add_line('\\end{document}', 'endings')
            