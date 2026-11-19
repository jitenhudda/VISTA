"""VISTA v4 — wide, beginner-friendly document dashboard and robust code scanner."""
from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import re

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from vista_core import (
    BASE, MODEL_FILE, answer_question, decode_codes, delete_history,
    detect_document_type, detect_language, extract_details, explain_document,
    list_history, ocr_availability, read_document, save_history, speech_mp3,
    summarize_text, translate_text, validate_document,
)

st.set_page_config(page_title='VISTA | Document Intelligence Studio', page_icon='✦', layout='wide', initial_sidebar_state='collapsed')
st.markdown(''' 
<style>
/* VISTA Studio: premium but simple. Full-width results, clean typography. */
.stApp{background:#f5f7fc;color:#132746}
.block-container{max-width:1540px;padding:1.4rem 2.1rem 3rem}
[data-testid="stHeader"]{background:#f5f7fc}
h1,h2,h3{font-family:inherit;color:#122c51;letter-spacing:-.023em}
h2{font-size:1.55rem;font-weight:750}
h3{font-size:1.22rem;font-weight:740}
[data-testid="stMetric"]{padding:18px 20px!important;border:1px solid #dce6f4;border-radius:18px;background:#fff;box-shadow:0 7px 27px rgba(21,54,92,.035)}
[data-testid="stMetricLabel"]{color:#657992;font-size:.87rem}
[data-testid="stMetricValue"]{font-size:1.5rem;letter-spacing:-.025em}
.stButton>button,.stDownloadButton>button{border-radius:12px;min-height:3rem;font-weight:650}
.stButton>button[kind="primary"]{background:#255de3;border:1px solid #255de3;color:#fff}
.stButton>button:hover{border-color:#255de3}
.stTabs [data-baseweb="tab-list"]{gap:12px;flex-wrap:wrap;border-bottom:1px solid #dce5f2}
.stTabs [data-baseweb="tab"]{font-weight:680;padding:13px 15px;border-radius:10px 10px 0 0}
.stMarkdown p{line-height:1.65}
.stTextArea textarea{line-height:1.68!important;font-size:1.06rem!important;border-radius:14px!important;padding:15px 18px!important;border-color:#d7e2ef!important}
.stTextInput input{font-size:1rem}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px!important}
[data-testid="stCameraInput"] video,[data-testid="stCameraInput"] img{max-height:360px;object-fit:contain}
.v-head{padding:28px 34px;border-radius:20px;background:linear-gradient(112deg,#10294f,#1f4e89 64%,#2665a7);color:white;margin:0 0 20px;box-shadow:0 15px 30px rgba(21,55,102,.1)}
.v-head h1{color:#fff;font-size:2.42rem;line-height:1.1;margin:6px 0 10px}
.v-head p{color:#ddeafa;margin:0;font-size:1.04rem}
.v-brow{color:#bfd4f3;font-weight:770;letter-spacing:.14em;font-size:.76rem}
.v-how{background:#fff;border:1px solid #e1e9f5;border-radius:14px;padding:16px 20px;color:#52647f;margin:14px 0 20px;font-size:.98rem}
.v-how strong{color:#285bd0}
.v-kicker{color:#2e60bb;font-weight:800;letter-spacing:.09em;font-size:.76rem;margin-top:15px}
.v-result{background:#eaf2ff;border:1px solid #d9e7ff;border-radius:12px;padding:12px 17px;color:#34567e;margin:9px 0 16px}
.v-qr-guide{background:#e9f3ff;border:1px solid #cfe1fc;border-radius:12px;padding:13px 16px;color:#34567e;margin:7px 0 16px}
@media(max-width:800px){.block-container{padding:.35rem .8rem 2rem}.v-head{padding:21px}.v-head h1{font-size:1.75rem}.stTextArea textarea{font-size:1rem!important}}
</style>
''', unsafe_allow_html=True)

st.markdown('''<section class="v-head"><div class="v-brow">DOCUMENT INTELLIGENCE · VISTA STUDIO</div>
<h1>Everything your document says, made simple.</h1>
<p>Scan text and QR codes · Identify documents · Understand, translate &amp; save results</p></section>''', unsafe_allow_html=True)

st.session_state.setdefault('workspace_name', 'Student')
st.session_state.setdefault('scan_generation', 0)


def load_image(raw: bytes):
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        return ImageOps.exif_transpose(img).convert('RGB')
    except Exception:
        return None


def get_sig(data: bytes, name: str, language: str, handwriting: bool) -> str:
    return hashlib.sha256(data + name.encode() + language.encode() + str(handwriting).encode()).hexdigest()


@st.cache_resource(show_spinner=False)
def get_model():
    if not MODEL_FILE.exists():
        return None
    try:
        import joblib
        model = joblib.load(MODEL_FILE)
        classifier = model.named_steps.get('classifier')
        if classifier is not None and not hasattr(classifier, 'multi_class'):
            classifier.multi_class = 'auto'
        model.predict(['Sample text'])
        return model
    except Exception:
        return None


def ml_predict(text: str):
    model = get_model()
    if model is None:
        return None
    lines = [x.strip() for x in text.splitlines() if x.strip()][:80]
    if not lines:
        return pd.DataFrame(columns=['Text', 'FUNSD field', 'Model score'])
    probs = model.predict_proba(lines)
    preds = probs.argmax(axis=1)
    return pd.DataFrame({
        'Text': lines,
        'FUNSD field': model.classes_[preds],
        'Model score': [f'{value:.1%}' for value in probs.max(axis=1)],
    })


scan_tab, qr_tab, insights_tab, history_tab = st.tabs([
    '1  Scan document', '2  QR & barcode', '3  Dashboard & ML', '4  History',
])

with scan_tab:
    st.markdown('<div class="v-how"><strong>Step 1</strong>  Upload a file or take a photo &nbsp; · &nbsp; <strong>Step 2</strong>  Click Scan document &nbsp; · &nbsp; <strong>Step 3</strong>  Read your results in the large panels below.</div>', unsafe_allow_html=True)
    source = st.radio('Choose how to scan', ['📁 Upload photo / PDF', '📷 Take camera photo', '🧪 Try sample'], horizontal=True)
    data, filename = None, None
    if source.startswith('📁'):
        upload = st.file_uploader('Upload your image or PDF', type=['jpg','jpeg','png','webp','pdf'], key='upload_v3')
        if upload:
            data, filename = upload.getvalue(), upload.name
    elif source.startswith('📷'):
        st.caption('Allow camera in Chrome, hold the whole page steady in good light, and press Take Photo.')
        camera = st.camera_input('Take Photo', key='camera_v3')
        if camera:
            data, filename = camera.getvalue(), 'camera_photo.jpg'
    else:
        samples = sorted((BASE / 'data' / 'FUNSD' / 'testing_data' / 'images').glob('*.png'))
        if samples:
            data, filename = samples[0].read_bytes(), samples[0].name
            st.info('Selected a real FUNSD sample form. Click Scan document to test OCR.')
        else:
            st.warning('Sample image missing. Please upload your own file.')

    with st.expander('⚙ Optional settings — leave defaults for English', expanded=False):
        scan_language = st.selectbox('Text language in the image', ['English', 'Hindi', 'Spanish', 'Japanese'])
        handwritten = st.checkbox('Handwriting (optional EasyOCR installation required)')
        st.caption('For Hindi/Japanese use a matching Tesseract language pack or EasyOCR. RapidOCR fallback is most suitable for printed English.')

    ready = bool(data)
    scan_sig = get_sig(data, filename, scan_language, handwritten) if ready else ''
    go = st.button('✨ Scan document', type='primary', disabled=not ready, use_container_width=True)
    if not ready:
        st.caption('Select a document above to enable Scan document.')

    if go and ready:
        try:
            with st.spinner('Reading your file… Scanned pages may take a little longer.'):
                result = read_document(data, filename, scan_language, handwritten, max_pages=5)
                st.session_state['scan_generation'] += 1
                st.session_state['v3_scan'] = {'sig':scan_sig, 'filename':filename, **result}
                for key in ('v3_translation', 'v3_translation_sig', 'v3_answer', 'v3_ml', 'v3_audio'):
                    st.session_state.pop(key, None)
            if result['text'].strip():
                st.success('Document scanned successfully!')
            else:
                st.warning('The scan finished, but no readable text was detected. Try brighter lighting or a larger photo.')
        except Exception as error:
            st.session_state.pop('v3_scan', None)
            st.error(f'Unable to scan: {error}')
            st.info('OCR setup check is available under Dashboard & ML. Printed English images can use RapidOCR without installing a separate Tesseract executable.')

    result = st.session_state.get('v3_scan')
    if result and ready and scan_sig == result['sig']:
        st.divider()
        st.markdown('<div class="v-kicker">YOUR RESULTS</div>', unsafe_allow_html=True)
        st.subheader('Document overview')
        edit_key = f"v3_editor_{scan_sig}_{st.session_state['scan_generation']}"
        # Recover edits from session state before drawing result metrics.
        current_text = st.session_state.get(edit_key, result['text'])
        classification = detect_document_type(current_text)
        language = detect_language(current_text)
        count_words = len(current_text.split())
        a,b,c,d = st.columns([1.3,1.0,.85,.6],gap='small')
        a.metric('Document type', classification['type'])
        b.metric('Category', classification['category'])
        c.metric('Language', language)
        d.metric('Words', count_words)
        st.markdown('<div class="v-result">You can edit OCR mistakes below. The summary and document type use the corrected text.</div>',unsafe_allow_html=True)
        editor_col,preview_col = st.columns([1.7,1],gap='large')
        with editor_col:
            st.markdown('#### Recognized text — full view')
            text = st.text_area('Recognized text', value=result['text'], height=510, key=edit_key,
                                label_visibility='collapsed',
                                help='This is the full, editable output. You can copy it or download it.')
            st.download_button('Download full recognized text', text, file_name='vista_text.txt', mime='text/plain',key='v4_text_download')
        with preview_col:
            st.markdown('#### Original document')
            if filename.lower().endswith('.pdf'):
                try:
                    import fitz
                    with fitz.open(stream=data,filetype='pdf') as source_pdf:
                        page=source_pdf[0]
                        pix=page.get_pixmap(matrix=fitz.Matrix(1.1,1.1),alpha=False)
                        st.image(Image.frombytes('RGB',(pix.width,pix.height),pix.samples),use_container_width=True)
                    st.caption(f'PDF · {result["pages"]} page(s) · {result["processed_pages"]} processed')
                except Exception:
                    st.info(f'PDF · {result["pages"]} page(s) · {result["processed_pages"]} processed')
            else:
                preview = load_image(data)
                if preview is not None:
                    st.image(preview,use_container_width=True)
            if classification['evidence']:
                st.caption('Detected from words: '+', '.join(classification['evidence'][:4]))
            else:
                st.caption('Document type is a text-based estimate, not a trained visual ID verifier.')
        if result['pages'] > result['processed_pages']:
            st.warning(f'Analyzed {result["processed_pages"]} of {result["pages"]} pages. This quick scan has a 5-page limit.')
        st.markdown('### Understand your document')
        over, translate, ask, details = st.tabs(['📝 Summary', '🌐 Translate', '💬 Ask questions', '🔍 Details & ML'])
        with over:
            st.markdown('#### A simple explanation')
            with st.container(border=True):
                st.markdown(explain_document(text))
            st.markdown('#### Most important lines')
            st.text_area('Summary output',summarize_text(text,6),height=310,
                         key=f'v4_summary_{scan_sig}_{hashlib.sha256(text.encode()).hexdigest()[:10]}',disabled=True,label_visibility='collapsed')
            st.caption('Summary uses offline text rules. It is not generative AI.')
            save_label = st.session_state['workspace_name'].strip() or 'Student'
            save_hash = hashlib.sha256((scan_sig + text + save_label).encode()).hexdigest()
            if st.button('Save document to history', key='v3_save'):
                if st.session_state.get('v3_saved') == save_hash:
                    st.info('Already saved during this session.')
                else:
                    save_history(save_label, filename, classification['type'], text)
                    st.session_state['v3_saved'] = save_hash
                    st.success('Saved to History.')

        with translate:
            st.markdown('#### Translation — full reading view')
            st.caption('Large side-by-side reading panels. Free translation service may have rate limits; first 1,400 characters per request.')
            lang_target = st.selectbox('Choose output language', ['Hindi','English','Spanish','Japanese'], key='v3_target')
            if st.button('🌐 Translate now', type='primary', disabled=not bool(text.strip()), key='v3_translate'):
                try:
                    with st.spinner('Translating…'):
                        st.session_state['v3_translation'] = translate_text(text[:1400], lang_target)
                        st.session_state['v3_translation_sig'] = (scan_sig, hashlib.sha256(text.encode()).hexdigest(), lang_target)
                except Exception:
                    st.warning('Online translator is temporarily unavailable or has reached its rate limit. Other scanning features still work.')
            a,b = st.columns(2,gap='medium')
            with a:
                st.text_area('Original text', text[:1400], height=550, disabled=True)
            with b:
                show = ''
                if st.session_state.get('v3_translation_sig') == (scan_sig, hashlib.sha256(text.encode()).hexdigest(), lang_target):
                    show = st.session_state.get('v3_translation','')
                st.text_area(f'{lang_target} translation', show or 'Click “Translate now” to see the translation here.', height=550, disabled=True)
                if show:
                    st.download_button('Download translation', show, file_name='vista_translation.txt', mime='text/plain')
            st.caption('Translation requires internet. For long documents, translate smaller sections. Never rely on automated translation for legal or medical decisions.')

        with ask:
            st.markdown('#### Ask about the scanned document')
            with st.form('v3_qa_form'):
                query = st.text_input('Your question', placeholder='Example: What is the invoice number?')
                submitted = st.form_submit_button('Find an answer')
            if submitted:
                st.session_state['v3_answer'] = answer_question(text, query)
            if st.session_state.get('v3_answer'):
                st.info(st.session_state['v3_answer'])
            st.caption('Finds matching information in the text, not an online chatbot.')
            if st.button('🔊 Generate speech', disabled=not bool(text.strip()), key='v3_speech'):
                try:
                    with st.spinner('Generating audio…'):
                        st.session_state['v3_audio'] = speech_mp3(text[:1200], language)
                except Exception:
                    st.warning('Speech generation needs internet and could not connect right now.')
            if st.session_state.get('v3_audio'):
                st.audio(st.session_state['v3_audio'], format='audio/mp3')

        with details:
            st.markdown('#### Detected details and checks')
            fields = extract_details(text, classification['category'])
            if fields:
                for key,value in fields.items():
                    st.markdown(f'**{key}:** {value}')
            else:
                st.caption('No standard fields found in this document.')
            for message in validate_document(text,classification['category']):
                st.write('• '+message)
            st.caption('Document type detection uses OCR keyword rules, not an image classification dataset. Always verify IDs, totals and dates.')
            st.markdown('#### FUNSD ML field classifier')
            st.caption('This independently trained model detects Question, Answer, Header and Other fields; it does not classify invoice/ID images.')
            if st.button('Analyze fields with ML', key='v3_run_ml', disabled=not bool(text.strip())):
                st.session_state['v3_ml'] = ml_predict(text)
            frame = st.session_state.get('v3_ml')
            if frame is not None:
                if frame.empty:
                    st.info('No text lines to classify.')
                else:
                    st.dataframe(frame, use_container_width=True, hide_index=True)
                    counts = frame['FUNSD field'].value_counts().rename_axis('Field').reset_index(name='Lines')
                    st.bar_chart(counts.set_index('Field'), color='#3b65cf')
                    st.download_button('Download predictions CSV', frame.to_csv(index=False), file_name='vista_ml.csv')
            if MODEL_FILE.exists() is False:
                st.warning('Trained ML model not found; run train_model.py from VS Code.')
    elif result and ready:
        st.info('You changed the input or settings. Press Scan document to refresh the results.')

with qr_tab:
    st.subheader('Scan a QR code or barcode')
    st.markdown('<div class="v-qr-guide"><strong>Tip:</strong> Show the <strong>entire square QR code</strong> with some white margin on every side. Keep your phone or paper steady, and avoid glare. A cut-off QR cannot be decoded reliably.</div>',unsafe_allow_html=True)
    qr_source = st.radio('Choose code input', ['Upload photo / PDF','Use camera','Try demo QR','Try demo barcode'],horizontal=True,key='v4_qr_source')
    qr_bytes, qr_name = None, None
    if qr_source == 'Upload photo / PDF':
        qr_upload = st.file_uploader('Choose photo or PDF of the code',type=['jpg','jpeg','png','webp','pdf'],key='v4_qr_upload')
        if qr_upload:
            qr_bytes,qr_name=qr_upload.getvalue(),qr_upload.name
    elif qr_source == 'Use camera':
        cam = st.camera_input('Take a clear photo of the entire code',key='v4_qr_camera')
        if cam:
            qr_bytes,qr_name=cam.getvalue(),'camera_qr.jpg'
    else:
        demo_name='demo_qr.png' if qr_source=='Try demo QR' else 'demo_barcode.png'
        demo_path=BASE/'assets'/demo_name
        if demo_path.is_file():
            qr_bytes,qr_name=demo_path.read_bytes(),demo_name
            st.info('Demo image selected. Press the button below to test the scanner without using your camera.')
        else:
            st.error('Demo image is missing. Try uploading your own picture.')

    if qr_bytes:
        st.markdown('#### Code preview')
        try:
            if qr_name.lower().endswith('.pdf'):
                import fitz
                with fitz.open(stream=qr_bytes,filetype='pdf') as qr_pdf:
                    preview_pix=qr_pdf[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
                    preview_img=Image.frombytes('RGB',(preview_pix.width,preview_pix.height),preview_pix.samples)
            else:
                preview_img=load_image(qr_bytes)
            if preview_img is not None:
                with st.container(border=True):
                    col_pic,col_tip=st.columns([1.5,1],gap='large')
                    with col_pic:
                        st.image(preview_img,use_container_width=True)
                    with col_tip:
                        st.markdown('**Having trouble scanning?**')
                        st.write('Make sure all four corners are visible. Hold it closer, keep the full code sharp, and try the Top right / Center options below.')
        except Exception:
            st.warning('Could not render the preview.')
        qr_focus=st.selectbox('Where is the code in the image?',
            ['Automatic (recommended)','Whole picture','Center','Top left','Top right','Bottom left','Bottom right'],key='v4_qr_focus',
            help='Automatic checks the whole photo and five zoomed areas. Manually choose the area if needed.')
        code_sig=hashlib.sha256(qr_bytes+qr_focus.encode()).hexdigest()
        if st.button('Read QR / barcode', type='primary', key='v4_qr_read',use_container_width=True):
            try:
                with st.spinner('Looking for the code…'):
                    if qr_name.lower().endswith('.pdf'):
                        import fitz
                        with fitz.open(stream=qr_bytes,filetype='pdf') as doc:
                            frames=[]
                            for i in range(min(2,len(doc))):
                                pix=doc[i].get_pixmap(matrix=fitz.Matrix(1.8,1.8),alpha=False)
                                frames.append(Image.frombytes('RGB',(pix.width,pix.height),pix.samples))
                    else:
                        frames=[load_image(qr_bytes)]
                    codes,status=decode_codes([f for f in frames if f is not None],focus=qr_focus)
                    st.session_state['v4_qr_result']=(code_sig,codes,status)
            except Exception as exc:
                st.error(f'Scanner error: {exc}')
        previous=st.session_state.get('v4_qr_result')
        if previous is not None and previous[0] == code_sig:
            _,codes,status=previous
            if codes:
                st.success(f'Success — {len(codes)} code(s) detected')
                for i,(format_name,value) in enumerate(codes):
                    st.markdown(f'**{format_name} code**')
                    st.code(value,language=None)
                    st.download_button('Download decoded text',value,file_name=f'vista_code_{i+1}.txt',key=f'dl_code_{i}')
            else:
                st.warning('Code could not be decoded from this image. Try full code visibility, better focus, a new photo, or another region.')
            st.caption(status)
    else:
        st.caption('Choose an input above to begin. Demo QR and barcode are included for quick testing.')

with insights_tab:
    st.subheader('Dashboard & dataset insights')
    st.caption('Real FUNSD evaluation plus your own saved scans — no invented performance numbers.')
    stats_path=BASE/'models'/'evaluation.json'
    stats=json.loads(stats_path.read_text(encoding='utf8')) if stats_path.exists() else None
    train_count=len(list((BASE/'data'/'FUNSD'/'training_data'/'annotations').glob('*.json')))
    test_count=len(list((BASE/'data'/'FUNSD'/'testing_data'/'annotations').glob('*.json')))
    username=st.session_state['workspace_name'].strip() or 'Student'
    previous_scans=list_history(username)
    first,second,third,fourth=st.columns(4)
    first.metric('Training forms', train_count)
    second.metric('Test forms', test_count)
    third.metric('Field ML accuracy',f'{stats["accuracy"]:.1%}' if stats else 'Not trained')
    fourth.metric('Scans saved',len(previous_scans))

    col_a,col_b=st.columns(2,gap='large')
    with col_a:
        st.markdown('#### Your saved document types')
        if previous_scans:
            kind_counts=Counter(row[2] for row in previous_scans)
            chart=pd.DataFrame({'Document type':list(kind_counts.keys()),'Scans':list(kind_counts.values())})
            st.bar_chart(chart.set_index('Document type'),color='#305fe0')
        else:
            st.info('Scan and save documents to build this chart automatically.')
    with col_b:
        st.markdown('#### FUNSD model performance')
        if stats:
            per=stats['per_class']
            chart=pd.DataFrame({'Field':list(per),'F1 score':[round(per[k]['f1-score'],3) for k in per]})
            st.bar_chart(chart.set_index('Field'),color='#21a7a3')
            st.caption(f'Accuracy measured on {stats["test_fields"]} annotated fields from {stats["test_documents"]} held-out forms; NOT OCR or document-type accuracy.')
        else:
            st.info('Run train_model.py to generate measured ML results.')
    with st.expander('📑 Model report and confusion matrix'):
        if stats:
            st.dataframe(pd.DataFrame(stats['per_class']).T, use_container_width=True)
            st.dataframe(pd.DataFrame(stats['confusion_matrix'],index=stats['labels'],columns=stats['labels']), use_container_width=True)
        st.write('FUNSD dataset trains a field classifier: Question / Answer / Header / Other. Invoice/ID category detection is a separate transparent OCR keyword component.')
    with st.expander('🛠 OCR setup diagnostics'):
        available=ocr_availability()
        st.write('**OCR engine:** '+available)
        st.write('**Model file:** '+('Found' if MODEL_FILE.exists() else 'Missing'))
        try:
            import zxingcpp
            st.write('**Barcode scanner:** ZXing installed (QR + 1D codes)')
        except ImportError:
            st.warning('ZXing missing. Run pip install -r requirements.txt for Code128/EAN barcodes.')
        st.caption('RapidOCR is installed through requirements.txt and does not need tesseract.exe. Its first scan may need an internet connection to download model weights. For Hindi/Japanese OCR use language packs or optional EasyOCR.')
        st.code('.venv\\Scripts\\python.exe check_setup.py',language='powershell')
    st.caption('Note: document type accuracy has not been measured against a labeled image-classification dataset.')

with history_tab:
    st.subheader('My saved documents')
    st.text_input('Workspace name',key='workspace_name',help='This is only a local label. It is not a password-protected account.')
    username=st.session_state['workspace_name'].strip() or 'Student'
    records=list_history(username)
    if not records:
        st.info('Nothing saved yet. Scan a document and press Save to history on Summary.')
    else:
        frame=pd.DataFrame(records,columns=['ID','Filename','Type','Date','Text'])
        st.download_button('Export history to CSV',frame.to_csv(index=False),file_name='vista_history.csv',mime='text/csv')
        for row_id,filename,typ,date,content in records:
            with st.expander(f'📄 {filename} · {typ} · {date}'):
                st.text_area('Recognized text',content,height=155,key=f'history_{row_id}')
                left,right=st.columns(2)
                with left:
                    st.download_button('Download TXT',content,file_name=f'vista_scan_{row_id}.txt',key=f'down_{row_id}')
                with right:
                    if st.button('Delete',key=f'del_{row_id}'):
                        delete_history(username,row_id)
                        st.rerun()
    st.caption('Local SQLite storage is not encrypted. Avoid saving sensitive actual IDs or patient data in this student demo.')

st.divider()
st.caption('VISTA Studio v4 · Real FUNSD dataset · Local OCR · ZXing-powered barcode reader · Verify important results')
