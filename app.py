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
