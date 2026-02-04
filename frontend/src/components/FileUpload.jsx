import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

const ACCEPTED_TYPES = {
    'application/pdf': ['.pdf'],
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
}

const MAX_SIZE = 50 * 1024 * 1024 // 50MB

function FileUpload({ onFileSelect, selectedFile, onRemove }) {
    const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
        if (rejectedFiles.length > 0) {
            const error = rejectedFiles[0].errors[0]
            if (error.code === 'file-too-large') {
                alert('File is too large. Maximum size is 50MB.')
            } else if (error.code === 'file-invalid-type') {
                alert('Invalid file type. Please upload a PDF or DOCX file.')
            }
            return
        }

        if (acceptedFiles.length > 0) {
            onFileSelect(acceptedFiles[0])
        }
    }, [onFileSelect])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: ACCEPTED_TYPES,
        maxSize: MAX_SIZE,
        multiple: false
    })

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return `${bytes} B`
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    }

    if (selectedFile) {
        return (
            <div className="file-info">
                <div className="file-info__icon">{selectedFile.type === 'application/pdf' ? '📕' : '📘'}</div>
                <div className="file-info__details">
                    <div className="file-info__name">{selectedFile.name}</div>
                    <div className="file-info__size">{formatFileSize(selectedFile.size)}</div>
                </div>
                <button
                    className="file-info__remove"
                    onClick={onRemove}
                    title="Remove file"
                >
                    ✕
                </button>
            </div>
        )
    }

    return (
        <div
            {...getRootProps()}
            className={`upload-zone ${isDragActive ? 'upload-zone--active' : ''}`}
        >
            <input {...getInputProps()} />
            <div className="upload-zone__icon">📤</div>
            <h3 className="upload-zone__title">
                {isDragActive ? 'Drop Archive' : 'Select Research Paper'}
            </h3>
            <p className="upload-zone__text">
                PDF or Word documents up to 50MB
            </p>
            <div className="upload-zone__formats">
                <span className="upload-zone__format">PDF</span>
                <span className="upload-zone__format">DOCX</span>
            </div>
        </div>
    )
}

export default FileUpload
