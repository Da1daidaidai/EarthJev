"""Build prompts for region scoring and answer generation."""


def answer_system(row):
    """System turn of every answer: the answer format of the question as an instruction (Sec. III-D)."""
    if row.get('multi_choice'):  # XLRS-lite land-use questions with one or more correct options
        return 'Reply with only the letter(s) of the correct option(s), separated by spaces. Do not explain.'
    if row['is_mcq']:
        return 'Reply with only the letter of the correct option. Do not explain.'
    return 'Reply with only a single word or phrase. Do not explain or describe the images.'


def search_query(row):
    """The question the Yes/No evidence prompt asks about: the stem, then the options of a multiple-choice question.
    The pointer trust passes the row with its stem stripped of pointers."""
    return row['stem'] + '\n' + '\n'.join(row['choice_options']) if row['is_mcq'] else row['stem']


def evidence_question(query):
    """ZoomSearch's Yes/No question whether a view shows the evidence for `query` (the trust signal y_j, Sec. III-B)."""
    return (f'Question: {query}\n\nBased on the current image crop shown:\n'
            '- Is the target object/subject mentioned in the question present in this cropped region?\n'
            '- Can you see the relevant visual content needed to answer this question?\n\nAnswer: Yes or No.')


def position_hint():
    """ZoomSearch's position hint, the first line of the canvas answer prompt."""
    return ('The input image has been padded with gray borders. When answering questions, the VLM should take these '
            'gray padding areas into account\n')


def canvas_answer(row, hint):
    """Text of the canvas presentation: the position hint, then the full question."""
    return f"{hint}Question:\n{row['question']}"


def _box_1000(box, w, h):
    x1, y1, x2, y2 = box
    return f'[{round(x1 * 1000 / w)},{round(y1 * 1000 / h)},{round(x2 * 1000 / w)},{round(y2 * 1000 / h)}]'


def memory_answer(row, image_size, boxes, notes=None):
    """Text of the memory presentation: image 1 is the whole image, images 2.. are the views at their pixel boxes
    [x1, y1, x2, y2], given as normalized 0-1000 positions; a view's note (the pointer it shows) ends its line."""
    w, h = image_size
    notes = notes or [None] * len(boxes)
    lines = [f'Image {i}: {_box_1000(box, w, h)}' + (f' ({note})' if note else '')
             for i, (box, note) in enumerate(zip(boxes, notes), start=2)]
    return (f'Image 1 is the whole remote-sensing image ({w}x{h} pixels, downsampled). '
            f'Images 2-{len(boxes) + 1} are full-resolution zoomed views of regions of the same image; '
            'their positions in the whole image as normalized [x1,y1,x2,y2] in 0-1000 are:\n'
            + '\n'.join(lines) + '\nUse all images to answer.\n' + row['question'])


def memory_answer_multi(row, image_sizes, boxes_per_image):
    """memory_answer for a question on several images (XLRS-lite change detection): images 1..n are the whole images,
    then the views of image 1, of image 2, ...; each view line names its image and its position in it."""
    n, k, lines = len(image_sizes), len(image_sizes) + 1, []
    for i, ((w, h), boxes) in enumerate(zip(image_sizes, boxes_per_image), start=1):
        for box in boxes:
            lines.append(f'Image {k}: image {i} {_box_1000(box, w, h)}')
            k += 1
    sizes = ', '.join(f'{w}x{h}' for w, h in image_sizes)
    return (f'Images 1-{n} are the {n} whole remote-sensing images ({sizes} pixels, downsampled). '
            f'Images {n + 1}-{k - 1} are full-resolution zoomed views of regions of them; for each view, the whole '
            'image it comes from and its position in that image as normalized [x1,y1,x2,y2] in 0-1000 are:\n'
            + '\n'.join(lines) + '\nUse all images to answer.\n' + row['question'])
