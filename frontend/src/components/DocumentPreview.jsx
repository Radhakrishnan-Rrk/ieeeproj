import { useState, useEffect } from 'react'

function DocumentPreview({ jobId }) {
    const [previewUrl, setPreviewUrl] = useState(null)
    const [error, setError] = useState(null)

    useEffect(() => {
        if (jobId) {
            // Create preview URL
            setPreviewUrl(`/api/preview/${jobId}`)
        }
    }, [jobId])

    if (error) {
        return (
            <div className="preview">
                <div className="preview__container">
                    <div className="preview__placeholder">
                        <p>Unable to load preview</p>
                        <p style={{ fontSize: '0.875rem', color: 'var(--text-dim)' }}>{error}</p>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="preview">
            <h3 className="preview__title">Live Preview</h3>
            <div className="preview__container">
                {previewUrl ? (
                    <iframe
                        className="preview__iframe"
                        src={previewUrl}
                        title="IEEE Paper Preview"
                        onError={() => setError('Failed to load preview')}
                    />
                ) : (
                    <div className="preview__placeholder">
                        <p>Rendering documentation...</p>
                    </div>
                )}
            </div>
        </div>
    )
}

export default DocumentPreview
