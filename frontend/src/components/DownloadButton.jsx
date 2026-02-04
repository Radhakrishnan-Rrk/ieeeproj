import { useState } from 'react'

function DownloadButton({ jobId }) {
    const [isDownloading, setIsDownloading] = useState(false)

    const handleDownload = async () => {
        setIsDownloading(true)

        try {
            const response = await fetch(`/api/download/${jobId}`)

            if (!response.ok) {
                throw new Error('Download failed')
            }

            // Get filename from Content-Disposition header or use default
            const contentDisposition = response.headers.get('Content-Disposition')
            let filename = `ieee_paper_${jobId.slice(0, 8)}.pdf`

            if (contentDisposition) {
                const match = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/)
                if (match && match[1]) {
                    filename = match[1].replace(/['"]/g, '')
                }
            }

            // Create blob and download
            const blob = await response.blob()
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = filename
            document.body.appendChild(a)
            a.click()
            window.URL.revokeObjectURL(url)
            document.body.removeChild(a)
        } catch (err) {
            console.error('Download error:', err)
            alert('Failed to download file. Please try again.')
        } finally {
            setIsDownloading(false)
        }
    }

    return (
        <button
            className="btn btn--success"
            onClick={handleDownload}
            disabled={isDownloading || !jobId}
        >
            {isDownloading ? (
                <>
                    <div className="spinner"></div>
                    <span>Downloading...</span>
                </>
            ) : (
                <>
                    📥 Download IEEE Paper
                </>
            )}
        </button>
    )
}

export default DownloadButton
