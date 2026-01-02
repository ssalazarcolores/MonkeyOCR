import re

from loguru import logger

from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.language import detect_lang
from magic_pdf.libs.markdown_utils import ocr_escape_special_markdown_char
from magic_pdf.post_proc.para_split_v3 import ListLineTag


def __is_hyphen_at_line_end(line):
    """Check if a line ends with one or more letters followed by a hyphen.

    Args:
    line (str): The line of text to check.

    Returns:
    bool: True if the line ends with one or more letters followed by a hyphen, False otherwise.
    """
    # Use regex to check if the line ends with one or more letters followed by a hyphen
    return bool(re.search(r'[A-Za-z]+-\s*$', line))


# Variable global para recolectar metadata de headers durante el procesamiento
_headers_metadata = []


def reset_headers_metadata():
    """Resetea la lista de metadata de headers."""
    global _headers_metadata
    _headers_metadata = []


def get_headers_metadata():
    """Retorna la metadata de headers recolectada."""
    global _headers_metadata
    return _headers_metadata.copy()


def _extract_title_metadata(block, page_num=0):
    """
    Extrae metadata visual de un bloque de título para corrección posterior con LLM.

    Returns:
        dict con: text, font_height, bbox, page, block_bbox, position_ratio
    """
    from magic_pdf.config.ocr_content_type import ContentType

    title_text = ''
    span_heights = []
    span_bboxes = []

    for line in block.get('lines', []):
        for span in line.get('spans', []):
            if span.get('type') == ContentType.Text:
                title_text += span.get('content', '')
                bbox = span.get('bbox', [])
                if len(bbox) >= 4:
                    height = bbox[3] - bbox[1]
                    if height > 0:
                        span_heights.append(height)
                        span_bboxes.append(bbox)

    title_text = title_text.strip()
    avg_font_height = sum(span_heights) / len(span_heights) if span_heights else 0

    # Calcular bbox del bloque completo
    block_bbox = block.get('bbox', [])
    if not block_bbox and span_bboxes:
        # Calcular bbox desde spans
        x0 = min(b[0] for b in span_bboxes)
        y0 = min(b[1] for b in span_bboxes)
        x1 = max(b[2] for b in span_bboxes)
        y1 = max(b[3] for b in span_bboxes)
        block_bbox = [x0, y0, x1, y1]

    # Calcular posición relativa en la página
    page_size = block.get('page_size', [])
    position_ratio = 0.5  # default: medio de la página
    if page_size and len(page_size) >= 2 and page_size[1] > 0 and block_bbox and len(block_bbox) >= 4:
        # Posición vertical relativa (0=top, 1=bottom)
        position_ratio = block_bbox[1] / page_size[1]

    return {
        'text': title_text,
        'font_height': round(avg_font_height, 2),
        'block_bbox': block_bbox,
        'page': page_num,
        'position_ratio': round(position_ratio, 3),
        'page_size': page_size
    }


def ocr_mk_mm_markdown_with_para_and_pagination(pdf_info_dict: list,
                                                img_buket_path,
                                                collect_headers=True):
    """
    Genera markdown con paginación y opcionalmente recolecta metadata de headers.

    Args:
        pdf_info_dict: Lista de información de páginas del PDF
        img_buket_path: Ruta base para imágenes
        collect_headers: Si True, recolecta metadata de headers (usar get_headers_metadata() después)
    """
    if collect_headers:
        reset_headers_metadata()

    markdown_with_para_and_pagination = []
    page_no = 0
    for page_info in pdf_info_dict:
        paras_of_layout = page_info.get('para_blocks')
        if not paras_of_layout:
            markdown_with_para_and_pagination.append({
                'page_no':
                    page_no,
                'md_content':
                    '',
            })
            page_no += 1
            continue
        page_markdown = ocr_mk_markdown_with_para_core_v2(
            paras_of_layout, 'mm', img_buket_path,
            page_num=page_no, collect_headers=collect_headers)
        markdown_with_para_and_pagination.append({
            'page_no':
                page_no,
            'md_content':
                '\n\n'.join(page_markdown)
        })
        page_no += 1
    return markdown_with_para_and_pagination


def ocr_mk_markdown_with_para_core_v2(paras_of_layout,
                                      mode,
                                      img_buket_path='',
                                      page_num=0,
                                      collect_headers=True,
                                      ):
    """
    Procesa bloques de párrafos y genera markdown.

    Args:
        paras_of_layout: Lista de bloques de párrafos
        mode: 'mm' para multimedia, 'nlp' para solo texto
        img_buket_path: Ruta base para imágenes
        page_num: Número de página actual (para metadata de headers)
        collect_headers: Si True, recolecta metadata de headers en _headers_metadata
    """
    global _headers_metadata

    page_markdown = []
    for para_block in paras_of_layout:
        para_text = ''
        para_type = para_block['type']
        if para_type in [BlockType.Text, BlockType.List, BlockType.Index]:
            para_text = merge_para_with_text(para_block)
        elif para_type == BlockType.Title:
            title_level = get_title_level(para_block)
            title_text = merge_para_with_text(para_block)

            # Nivel 0 = no es un header real, tratar como texto normal
            if title_level == 0:
                para_text = title_text  # Sin marcado de header
            else:
                para_text = f'{"#" * title_level} {title_text}'.replace('\n', f'\n{"#" * title_level}')

            # Recolectar metadata del header para corrección posterior con LLM
            if collect_headers:
                metadata = _extract_title_metadata(para_block, page_num)
                metadata['level_heuristic'] = title_level
                metadata['markdown'] = para_text.split('\n')[0] if title_level > 0 else ''
                _headers_metadata.append(metadata)
        elif para_type == BlockType.InterlineEquation:
            para_text = merge_para_with_text(para_block)
        elif para_type == BlockType.Image:
            if mode == 'nlp':
                continue
            elif mode == 'mm':
                for block in para_block['blocks']:
                    if block['type'] == BlockType.ImageBody:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.Image:
                                    if span.get('image_path', ''):
                                        para_text += f"\n![]({join_path(img_buket_path, span['image_path'])})  \n"
                for block in para_block['blocks']:
                    if block['type'] == BlockType.ImageCaption:
                        para_text += merge_para_with_text(block) + '  \n'
                for block in para_block['blocks']:
                    if block['type'] == BlockType.ImageFootnote:
                        para_text += merge_para_with_text(block) + '  \n'
        elif para_type == BlockType.Table:
            if mode == 'nlp':
                continue
            elif mode == 'mm':
                for block in para_block['blocks']:
                    if block['type'] == BlockType.TableCaption:
                        para_text += merge_para_with_text(block) + '  \n'
                for block in para_block['blocks']:
                    if block['type'] == BlockType.TableBody:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.Table:
                                    # if processed by table model
                                    if span.get('latex', ''):
                                        para_text += f"\n\n$\n {span['latex']}\n$\n\n"
                                    elif span.get('html', ''):
                                        para_text += f"\n\n{span['html']}\n\n"
                                    elif span.get('image_path', ''):
                                        para_text += f"\n![]({join_path(img_buket_path, span['image_path'])})  \n"
                for block in para_block['blocks']:
                    if block['type'] == BlockType.TableFootnote:
                        para_text += merge_para_with_text(block) + '  \n'

        if para_text.strip() == '':
            continue
        else:
            page_markdown.append(para_text.strip() + '  ')

    return page_markdown


def detect_language(text):
    en_pattern = r'[a-zA-Z]+'
    en_matches = re.findall(en_pattern, text)
    en_length = sum(len(match) for match in en_matches)
    if len(text) > 0:
        if en_length / len(text) >= 0.5:
            return 'en'
        else:
            return 'unknown'
    else:
        return 'empty'


def merge_para_with_text(para_block):
    block_text = ''
    for line in para_block['lines']:
        for span in line['spans']:
            if span['type'] in [ContentType.Text]:
                block_text += span['content']
    block_lang = detect_lang(block_text[:100])

    para_text = ''
    for i, line in enumerate(para_block['lines']):

        if i >= 1 and line.get(ListLineTag.IS_LIST_START_LINE, False):
            para_text += '  \n'

        for j, span in enumerate(line['spans']):

            span_type = span['type']
            content = ''
            if span_type == ContentType.Text:
                content = ocr_escape_special_markdown_char(span['content'])
            elif span_type == ContentType.InlineEquation:
                content = f"${span['content']}$"
            elif span_type == ContentType.InterlineEquation:
                content = f"\n$$\n{span['content']}\n$$\n"

            content = content.strip()

            if content:
                langs = ['zh', 'ja', 'ko']
                # logger.info(f'block_lang: {block_lang}, content: {content}')
                if block_lang in langs: # In Chinese/Japanese/Korean context, line breaks don't need space separation, but if it's inline equation ending, still need to add space
                    if j == len(line['spans']) - 1 and span_type not in [ContentType.InlineEquation]:
                        para_text += content
                    else:
                        para_text += f'{content} '
                else:
                    if span_type in [ContentType.Text, ContentType.InlineEquation]:
                        # If span is last in line and ends with hyphen, no space should be added at end, and hyphen should be removed
                        if j == len(line['spans'])-1 and span_type == ContentType.Text and __is_hyphen_at_line_end(content):
                            para_text += content[:-1]
                        else:  # In Western text context, content needs space separation
                            para_text += f'{content} '
                    elif span_type == ContentType.InterlineEquation:
                        para_text += content
            else:
                continue
    # Split connected characters
    # para_text = __replace_ligatures(para_text)

    return para_text


def para_to_standard_format_v2(para_block, img_buket_path, page_idx, drop_reason=None):
    para_type = para_block['type']
    para_content = {}
    if para_type in [BlockType.Text, BlockType.List, BlockType.Index]:
        para_content = {
            'type': 'text',
            'text': merge_para_with_text(para_block),
        }
    elif para_type == BlockType.Title:
        title_level = get_title_level(para_block)
        para_content = {
            'type': 'text',
            'text': merge_para_with_text(para_block),
            'text_level': title_level,
        }
    elif para_type == BlockType.InterlineEquation:
        para_content = {
            'type': 'equation',
            'text': merge_para_with_text(para_block),
            'text_format': 'latex',
        }
    elif para_type == BlockType.Image:
        para_content = {'type': 'image', 'img_path': '', 'img_caption': [], 'img_footnote': []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.ImageBody:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.Image:
                            if span.get('image_path', ''):
                                para_content['img_path'] = join_path(img_buket_path, span['image_path'])
            if block['type'] == BlockType.ImageCaption:
                para_content['img_caption'].append(merge_para_with_text(block))
            if block['type'] == BlockType.ImageFootnote:
                para_content['img_footnote'].append(merge_para_with_text(block))
    elif para_type == BlockType.Table:
        para_content = {'type': 'table', 'img_path': '', 'table_caption': [], 'table_footnote': []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.TableBody:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.Table:

                            if span.get('latex', ''):
                                para_content['table_body'] = f"\n\n$\n {span['latex']}\n$\n\n"
                            elif span.get('html', ''):
                                para_content['table_body'] = f"\n\n{span['html']}\n\n"

                            if span.get('image_path', ''):
                                para_content['img_path'] = join_path(img_buket_path, span['image_path'])

            if block['type'] == BlockType.TableCaption:
                para_content['table_caption'].append(merge_para_with_text(block))
            if block['type'] == BlockType.TableFootnote:
                para_content['table_footnote'].append(merge_para_with_text(block))

    para_content['page_idx'] = page_idx

    if drop_reason is not None:
        para_content['drop_reason'] = drop_reason

    return para_content


def union_make(pdf_info_dict: list,
               make_mode: str,
               drop_mode: str,
               img_buket_path: str = '',
               collect_headers: bool = True,
               ):
    """
    Función principal para generar markdown desde PDF.

    Args:
        pdf_info_dict: Información del PDF
        make_mode: Modo de generación (MM_MD, NLP_MD, STANDARD_FORMAT)
        drop_mode: Modo de manejo de páginas a descartar
        img_buket_path: Ruta base para imágenes
        collect_headers: Si True, recolecta metadata de headers (usar get_headers_metadata() después)
    """
    if collect_headers:
        reset_headers_metadata()

    output_content = []
    for page_info in pdf_info_dict:
        drop_reason_flag = False
        drop_reason = None
        if page_info.get('need_drop', False):
            drop_reason = page_info.get('drop_reason')
            if drop_mode == DropMode.NONE:
                pass
            elif drop_mode == DropMode.NONE_WITH_REASON:
                drop_reason_flag = True
            elif drop_mode == DropMode.WHOLE_PDF:
                raise Exception((f'drop_mode is {DropMode.WHOLE_PDF} ,'
                                 f'drop_reason is {drop_reason}'))
            elif drop_mode == DropMode.SINGLE_PAGE:
                logger.warning((f'drop_mode is {DropMode.SINGLE_PAGE} ,'
                                f'drop_reason is {drop_reason}'))
                continue
            else:
                raise Exception('drop_mode can not be null')

        paras_of_layout = page_info.get('para_blocks')
        page_idx = page_info.get('page_idx', 0)
        if not paras_of_layout:
            continue
        if make_mode == MakeMode.MM_MD:
            page_markdown = ocr_mk_markdown_with_para_core_v2(
                paras_of_layout, 'mm', img_buket_path,
                page_num=page_idx, collect_headers=collect_headers)
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.NLP_MD:
            page_markdown = ocr_mk_markdown_with_para_core_v2(
                paras_of_layout, 'nlp',
                page_num=page_idx, collect_headers=collect_headers)
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.STANDARD_FORMAT:
            for para_block in paras_of_layout:
                if drop_reason_flag:
                    para_content = para_to_standard_format_v2(
                        para_block, img_buket_path, page_idx)
                else:
                    para_content = para_to_standard_format_v2(
                        para_block, img_buket_path, page_idx)
                output_content.append(para_content)
    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        return '\n\n'.join(output_content)
    elif make_mode == MakeMode.STANDARD_FORMAT:
        return output_content


def get_title_level(block):
    """
    Determina el nivel de un título (0-4) para generar markdown.

    Heurísticas para papers científicos:
    - Nivel 0: NO son headers (Figure X, Table X, captions, metadata, ruido)
    - Nivel 1: Secciones principales (Abstract, I. Introduction, 1. Methods, etc.)
    - Nivel 2: Subsecciones (1.1, 2.1, II.A, II.1, A., Problem Statement, etc.)
    - Nivel 3: Sub-subsecciones (1.1.1, 1), 2), Definition 1, Theorem 2, etc.)
    - Nivel 4: Sub-sub-subsecciones (1.1.1.1, muy raras)

    También usa información visual del bbox para inferir tamaño de fuente.
    """
    import re

    # Si el bloque ya tiene nivel asignado (no es el default 1), usarlo
    if 'level' in block and block['level'] != 1:
        level = block['level']
        return max(0, min(4, level))

    # Extraer texto del título y calcular altura promedio de spans
    title_text = ''
    span_heights = []
    for line in block.get('lines', []):
        for span in line.get('spans', []):
            if span.get('type') == ContentType.Text:
                title_text += span.get('content', '')
                # Calcular altura del span desde bbox
                bbox = span.get('bbox', [])
                if len(bbox) >= 4:
                    height = bbox[3] - bbox[1]
                    if height > 0:
                        span_heights.append(height)

    title_text = title_text.strip()

    # Calcular altura promedio de fuente (inferida del bbox)
    avg_font_height = sum(span_heights) / len(span_heights) if span_heights else 0

    if not title_text:
        return 1

    title_lower = title_text.lower().strip()

    # === NIVEL 0: NO SON HEADERS REALES (eliminar) ===

    not_headers_patterns = [
        # Captions de figuras y tablas
        r'^fig(ure|\.)\s*\d+',           # Figure 1, Fig. 1, Figure 2:
        r'^figura\s*\d+',                 # Figura 1
        r'^tab(le|\.)\s*\d+',             # Table 1, Tab. 1
        r'^tabla\s*\d+',                  # Tabla 1
        r'^chart\s*\d+',                  # Chart 1
        r'^graph\s*\d+',                  # Graph 1
        r'^plate\s*\d+',                  # Plate 1
        r'^listing\s*\d+',                # Listing 1 (código)
        r'^code\s*\d+',                   # Code 1
        r'^equation\s*\d+',               # Equation 1
        r'^eq\.\s*\d+',                   # Eq. 1

        # Metadata y ruido de papers
        r'^©',                             # Copyright
        r'^\(c\)\s*\d{4}',                # (c) 2024
        r'^copyright',                     # Copyright notice
        r'^doi[:\s]',                      # DOI: 10.1234/...
        r'^https?://',                     # URLs
        r'^www\.',                         # www.example.com
        r'^arxiv[:\s]',                    # arXiv: 1234.5678
        r'^\d{4}\s+(ieee|acm|springer)',  # "2024 IEEE..."
        r'^ieee\s',                        # "IEEE Trans..."
        r'^acm\s',                         # "ACM ..."
        r'^springer\s',                    # "Springer ..."
        r'^elsevier\s',                    # "Elsevier ..."
        r'^issn[:\s]',                     # ISSN:
        r'^isbn[:\s]',                     # ISBN:
        r'^vol(ume)?\.?\s*\d+',           # Volume 1, Vol. 1
        r'^issue\s*\d+',                   # Issue 1
        r'^pp\.?\s*\d+',                   # pp. 1-10
        r'^pages?\s*\d+',                  # Page 1, Pages 1-10
        r'^received[:\s]',                 # Received: date
        r'^accepted[:\s]',                 # Accepted: date
        r'^published[:\s]',                # Published: date
        r'^revised[:\s]',                  # Revised: date
        r'^submitted[:\s]',                # Submitted: date
        r'^editor[:\s]',                   # Editor:
        r'^keywords?[:\s]',                # Keywords:
        r'^corresponding\s*author',        # Corresponding author
        r'^e-?mail[:\s]',                  # Email:, E-mail:
        r'^author\s*contributions?',       # Author contributions
        r'^conflict\s*of\s*interest',      # Conflict of interest
        r'^data\s*availability',           # Data availability
        r'^code\s*availability',           # Code availability
        r'^funding[:\s]',                  # Funding:
        r'^grant[:\s]',                    # Grant:
        r'^supported\s*by',                # Supported by
        r'^\*\s*(corresponding|equal)',    # * Corresponding author
        r'^†\s*equal',                     # † Equal contribution

        # Afiliaciones (detectar patrones comunes)
        r'^\d+\s*(department|school|university|institute|laboratory|lab|center|centre)',
        r'^(department|school|faculty)\s*of\s',

        # Headers de página/columna
        r'^page\s*\d+',                    # Page 1
        r'^\d+\s*of\s*\d+',                # 1 of 10
        r'^running\s*head',                # Running head

        # Fragmentos muy cortos que probablemente son ruido
        r'^[a-z]$',                        # Single lowercase letter
        r'^\d+$',                          # Solo números
        r'^[.\-_]+$',                      # Solo puntuación
    ]

    for pattern in not_headers_patterns:
        if re.match(pattern, title_lower):
            return 0  # Marcar para eliminar

    # === NIVEL 1: SECCIONES PRINCIPALES ===

    # Palabras clave que siempre son nivel 1
    level1_keywords = [
        # Estructura estándar de papers
        r'^abstract$',
        r'^resumen$',
        r'^references?$',
        r'^bibliograf[ií]a$',
        r'^acknowledg[e]?ments?$',
        r'^agradecimientos$',
        r'^appendix',
        r'^ap[ée]ndice',
        r'^supplementary',
        r'^conclusion[s]?$',
        r'^discussion$',
        r'^introduction$',
        r'^methods?$',
        r'^methodology$',
        r'^results?$',
        r'^related\s*work',
        r'^background$',
        r'^preliminaries$',
        r'^overview$',
        r'^motivation$',
        r'^summary$',
        # Índices y glosarios
        r'^index$',
        r'^glossary$',
        r'^nomenclature$',
        r'^abbreviations?$',
        r'^notation$',
    ]

    for pattern in level1_keywords:
        if re.match(pattern, title_lower):
            return 1

    # === NIVEL 2: SUBSECCIONES COMUNES ===

    # Palabras clave que típicamente son nivel 2 (subsecciones)
    level2_keywords = [
        # Metodología
        r'^problem\s*(statement|formulation|definition)',
        r'^experimental\s*(setup|design|results|evaluation)',
        r'^implementation\s*(details?)?',
        r'^data\s*(collection|analysis|description|preprocessing)',
        r'^evaluation\s*(metrics?|criteria|protocol)',
        r'^training\s*(details?|procedure|setup)',
        r'^model\s*(architecture|description|overview)',
        r'^network\s*architecture',
        r'^loss\s*function',
        r'^objective\s*function',
        r'^optimization',
        # Análisis
        r'^ablation\s*(study|analysis|experiments?)',
        r'^sensitivity\s*analysis',
        r'^comparative\s*(study|analysis)',
        r'^quantitative\s*(results?|analysis|evaluation)',
        r'^qualitative\s*(results?|analysis|evaluation)',
        r'^statistical\s*analysis',
        r'^error\s*analysis',
        # Limitaciones y futuro
        r'^limitations?$',
        r'^future\s*work',
        r'^ethical\s*(considerations?|statement)',
        r'^broader\s*impact',
        # Otros
        r'^contributions?$',
        r'^overview$',
        r'^notation$',
        r'^definitions?$',
        r'^assumptions?$',
        r'^setup$',
        r'^datasets?$',
        r'^baselines?$',
        r'^metrics?$',
        r'^hyperparameters?$',
        r'^parameter\s*settings?',
    ]

    for pattern in level2_keywords:
        if re.match(pattern, title_lower):
            return 2

    # === PATRONES DE NUMERACIÓN ===

    # Símbolo de sección § (papers europeos/matemáticos)
    section_symbol_patterns = [
        (r'^§\s*\d+\.\d+\.\d+', 3),      # §1.2.3
        (r'^§\s*\d+\.\d+', 2),            # §1.2
        (r'^§\s*\d+', 1),                 # §1
    ]

    for pattern, level in section_symbol_patterns:
        if re.match(pattern, title_text):
            return level

    # Numeración decimal (más común en papers)
    # Más restrictivo: requiere que después del número haya punto o texto
    decimal_patterns = [
        (r'^\d+\.\d+\.\d+\.\d+[\.\s]', 4),   # 1.2.3.4. o 1.2.3.4 texto
        (r'^\d+\.\d+\.\d+[\.\s]', 3),         # 1.2.3. o 1.2.3 texto
        (r'^\d+\.\d+[\.\s]', 2),               # 1.2. o 1.2 texto
        (r'^\d+\.\s+[A-Z]', 1),                # 1. Texto (sección principal)
        (r'^\d+\s+[A-Z]', 1),                  # 1 Texto (sin punto, también válido)
    ]

    for pattern, level in decimal_patterns:
        if re.match(pattern, title_text):
            return level

    # Numeración romana (IEEE style)
    roman_patterns = [
        (r'^[IVX]+\.[A-Z]\.\d+', 3),      # II.A.1
        (r'^[IVX]+\.[A-Z]\.?[\s\.]', 2),  # II.A o II.A.
        (r'^[IVX]+\.\d+', 2),              # II.1
        (r'^[IVX]+\.?\s+[A-Z]', 1),        # II. Texto o II Texto
    ]

    for pattern, level in roman_patterns:
        if re.match(pattern, title_text):
            return level

    # Letras mayúsculas (subsecciones)
    letter_upper_patterns = [
        (r'^[A-Z]\.\d+[\.\s]', 2),         # A.1. o A.1 texto
        (r'^\([A-Z]\)\s', 2),               # (A) texto
        (r'^[A-Z]\.?\s+[A-Z]', 2),          # A. Texto o A Texto
    ]

    for pattern, level in letter_upper_patterns:
        if re.match(pattern, title_text):
            return level

    # Letras minúsculas y números con paréntesis (sub-subsecciones)
    paren_patterns = [
        (r'^\d+\)\s', 3),                   # 1) texto
        (r'^\(\d+\)\s', 3),                 # (1) texto
        (r'^[a-z]\)\s', 3),                 # a) texto
        (r'^\([a-z]\)\s', 3),               # (a) texto
        (r'^[ivx]+\)\s', 3),                # i) ii) iii) (romanos minúscula)
        (r'^\([ivx]+\)\s', 3),              # (i) (ii) (iii)
    ]

    for pattern, level in paren_patterns:
        if re.match(pattern, title_text, re.IGNORECASE):
            return level

    # === ESTRUCTURAS MATEMÁTICAS/FORMALES (nivel 3) ===

    formal_structure_patterns = [
        # Teoremas y similares
        r'^theorem\s*\d*',
        r'^teorema\s*\d*',
        r'^lemma\s*\d*',
        r'^lema\s*\d*',
        r'^corollary\s*\d*',
        r'^corolario\s*\d*',
        r'^proposition\s*\d*',
        r'^proposici[oó]n\s*\d*',
        r'^conjecture\s*\d*',
        r'^conjetura\s*\d*',
        r'^claim\s*\d*',
        r'^property\s*\d*',
        r'^propiedad\s*\d*',
        # Definiciones y ejemplos
        r'^definition\s*\d*',
        r'^definici[oó]n\s*\d*',
        r'^example\s*\d*',
        r'^ejemplo\s*\d*',
        r'^remark\s*\d*',
        r'^observaci[oó]n\s*\d*',
        r'^note\s*\d*',
        r'^nota\s*\d*',
        # Algoritmos y procedimientos
        r'^algorithm\s*\d*',
        r'^algoritmo\s*\d*',
        r'^procedure\s*\d*',
        r'^procedimiento\s*\d*',
        r'^protocol\s*\d*',
        r'^protocolo\s*\d*',
        # Pruebas
        r'^proof',
        r'^demostraci[oó]n',
        r'^prueba',
        # Pasos y casos
        r'^step\s*\d+',
        r'^paso\s*\d+',
        r'^stage\s*\d+',
        r'^etapa\s*\d+',
        r'^phase\s*\d+',
        r'^fase\s*\d+',
        r'^case\s*\d+',
        r'^caso\s*\d+',
        r'^case\s+[ivx]+',                  # Case i, Case ii
        # Nota: Figure/Table ahora se detectan como nivel 0 (no headers)
    ]

    for pattern in formal_structure_patterns:
        if re.match(pattern, title_lower):
            return 3

    # === DETECCIÓN POR CARACTERÍSTICAS DEL TEXTO ===

    # Si el título es muy corto (1-3 palabras) y empieza con mayúscula,
    # probablemente es una sección principal
    words = title_text.split()
    if len(words) <= 3 and title_text[0].isupper():
        # Verificar que no sea una oración (no termina en punto o tiene verbos comunes)
        if not title_text.endswith('.') and not any(v in title_lower for v in [' is ', ' are ', ' was ', ' were ', ' the ', ' a ']):
            return 1

    # === DETECCIÓN POR TAMAÑO DE FUENTE (inferido de bbox) ===

    # Usar la altura del bbox como proxy del tamaño de fuente
    # Heurísticas basadas en papers típicos (Letter/A4, ~300 DPI):
    # - Títulos principales (nivel 1): altura > 18 px
    # - Subsecciones (nivel 2): altura 14-18 px
    # - Sub-subsecciones (nivel 3): altura 10-14 px
    # - Texto normal: altura < 12 px

    # También considerar el tamaño relativo a la página si está disponible
    page_size = block.get('page_size', [])
    if page_size and len(page_size) >= 2 and page_size[1] > 0:
        # Normalizar altura respecto a la página (típicamente ~800-1000 px)
        relative_height = avg_font_height / page_size[1]
        # Títulos muy grandes (>2.5% de la página) son nivel 1
        if relative_height > 0.025:
            return 1
        # Títulos grandes (1.8-2.5% de la página) son nivel 2
        elif relative_height > 0.018:
            return 2
        # Títulos medianos (1.2-1.8% de la página) son nivel 3
        elif relative_height > 0.012:
            return 3

    # Fallback: usar altura absoluta si no hay page_size
    if avg_font_height > 0:
        if avg_font_height > 22:
            return 1  # Fuente muy grande = título principal
        elif avg_font_height > 16:
            return 2  # Fuente grande = subsección
        elif avg_font_height > 12:
            return 3  # Fuente mediana = sub-subsección

    # Default: nivel 1 (título del paper o sección sin numerar)
    return 1