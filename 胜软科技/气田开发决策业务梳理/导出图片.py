import zipfile
import os
import xml.etree.ElementTree as ET
import openpyxl
import re

def sanitize_filename(filename):
    """Remove or replace invalid characters in a filename."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def read_excel_data(sheet):
    """Extract DISPIMG IDs and corresponding names from a worksheet."""
    image_mapping = {}
    image_list = []

    for row in sheet.iter_rows(min_row=1, values_only=False):
        for col_idx, cell in enumerate(row):
            if cell.value and isinstance(cell.value, str) and '=_xlfn.DISPIMG(' in cell.value:
                formula = cell.value
                start = formula.find('"') + 1
                end = formula.find('"', start)
                image_id = formula[start:end]

                # Get the content two columns to the left of the current cell
                name_cell = row[col_idx - 2].value if col_idx >= 2 else None
                if name_cell:
                    clean_name = sanitize_filename(str(name_cell).strip())
                    image_mapping[image_id] = clean_name
                    image_list.append(image_id)

    return image_mapping, image_list

def get_xml_id_image_map(xlsx_file_path):
    """Map DISPIMG IDs to image file paths in the XLSX archive."""
    with zipfile.ZipFile(xlsx_file_path, 'r') as zfile:
        with zfile.open('xl/cellimages.xml') as file:
            xml_content = file.read()
        with zfile.open('xl/_rels/cellimages.xml.rels') as file:
            relxml_content = file.read()

    root = ET.fromstring(xml_content)
    namespaces = {
        'xdr': 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing',
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'
    }

    name_to_embed_map = {}
    for pic in root.findall('.//xdr:pic', namespaces=namespaces):
        name = pic.find('.//xdr:cNvPr', namespaces=namespaces).attrib['name']
        embed = pic.find('.//a:blip', namespaces=namespaces).attrib[
            '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed']
        name_to_embed_map[name] = embed

    root1 = ET.fromstring(relxml_content)
    namespaces = {'r': 'http://schemas.openxmlformats.org/package/2006/relationships'}

    id_target_map = {child.attrib['Id']: child.attrib.get('Target', 'No Target Found') for child in
                     root1.findall('.//r:Relationship', namespaces=namespaces)}

    name_to_target_map = {name: id_target_map[embed] for name, embed in name_to_embed_map.items() if
                          embed in id_target_map}
    return name_to_target_map

def output_id_image(xlsx_file_path):
    """Extract images for each worksheet and save them in separate directories."""
    workbook = openpyxl.load_workbook(xlsx_file_path, data_only=False)
    name_to_target_map = get_xml_id_image_map(xlsx_file_path)

    # Create a directory for the workbook
    workbook_name = os.path.splitext(os.path.basename(xlsx_file_path))[0]
    workbook_directory = f'./{sanitize_filename(workbook_name)}_images'
    if not os.path.exists(workbook_directory):
        os.makedirs(workbook_directory)

    for sheet in workbook.worksheets:
        image_mapping, image_list = read_excel_data(sheet)
        new_map = {key: name_to_target_map.get(key) for key in image_list if key in name_to_target_map}

        # Create a directory for the worksheet
        sheet_name = sanitize_filename(sheet.title)
        sheet_directory = os.path.join(workbook_directory, sheet_name)
        if not os.path.exists(sheet_directory):
            os.makedirs(sheet_directory)

        with zipfile.ZipFile(xlsx_file_path, 'r') as zfile:
            for key, image_path in new_map.items():
                actual_image_path = f'xl/{image_path}'
                if actual_image_path in zfile.namelist():
                    with zfile.open(actual_image_path) as image_file:
                        image_content = image_file.read()
                        filename = sanitize_filename(image_mapping.get(key, key))  # Fallback to key if no name found
                        new_file_path = os.path.join(sheet_directory, f"{filename}.png")
                        with open(new_file_path, 'wb') as new_file:
                            new_file.write(image_content)
                else:
                    print(f"File {actual_image_path} not found in the archive.")

if __name__ == '__main__':
    # Specify the workbook to process
    excel_file = "安全环保IPOM表-20250224-黄.xlsx"
    if os.path.exists(excel_file):
        output_id_image(excel_file)
        print(f"图片已保存到以 {os.path.splitext(os.path.basename(excel_file))[0]} 命名的文件夹中")
    else:
        print(f"文件 {excel_file} 不存在，请检查文件名并重试。")