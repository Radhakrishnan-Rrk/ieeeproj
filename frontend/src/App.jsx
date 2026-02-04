import { useState, useCallback, useEffect } from 'react'
import FileUpload from './components/FileUpload'
import ConversionProgress from './components/ConversionProgress'
import DocumentPreview from './components/DocumentPreview'
import DownloadButton from './components/DownloadButton'

const API_BASE = '/api'

function App() {
    const [file, setFile] = useState(null)
    const [jobId, setJobId] = useState(null)
    const [status, setStatus] = useState(null)
    const [error, setError] = useState(null)
    const [isUploading, setIsUploading] = useState(false)

    // Poll for status updates
    useEffect(() => {
        if (!jobId || status?.status === 'completed' || status?.status === 'failed') {
            return
        }

        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`${API_BASE}/status/${jobId}`)
                if (response.ok) {
                    const data = await response.json()
                    setStatus(data)

                    if (data.status === 'completed' || data.status === 'failed') {
                        clearInterval(pollInterval)
                    }

                    if (data.status === 'failed') {
                        setError(data.error || 'Conversion failed')
                    }
                }
            } catch (err) {
                console.error('Status poll error:', err)
            }
        }, 1000)

        return () => clearInterval(pollInterval)
    }, [jobId, status?.status])

    const handleFileSelect = useCallback((selectedFile) => {
        setFile(selectedFile)
        setError(null)
        setStatus(null)
        setJobId(null)
    }, [])

    const handleUpload = useCallback(async () => {
        if (!file) return

        setIsUploading(true)
        setError(null)

        try {
            const formData = new FormData()
            formData.append('file', file)

            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                const errorData = await response.json()
                throw new Error(errorData.detail || 'Upload failed')
            }

            const data = await response.json()
            setJobId(data.job_id)
            setStatus({
                status: 'pending',
                progress: 0,
                message: 'Starting conversion...'
            })
        } catch (err) {
            setError(err.message)
        } finally {
            setIsUploading(false)
        }
    }, [file])

    const handleReset = useCallback(() => {
        setFile(null)
        setJobId(null)
        setStatus(null)
        setError(null)
    }, [])

    const isConverting = status && ['pending', 'processing'].includes(status.status)
    const isCompleted = status?.status === 'completed'

    return (
        <div className="app">
            {/* Navbar */}
            <nav className="navbar">
                <div className="navbar__container">
                    <a href="/" className="navbar__logo">
                        <span className="navbar__logo-icon">📄</span>
                        IEEE CONVERTER
                    </a>
                    <div className="navbar__links">
                        <a href="#how" className="navbar__link">Process</a>
                        <a href="#about" className="navbar__link">About</a>
                        <a href="#login" className="navbar__login">Sign In</a>
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            {!isConverting && !isCompleted && (
                <header className="hero animate-fadeIn">
                    <div className="hero__badge">
                        <span>Academic Transformation</span>
                    </div>
                    <h1 className="hero__title">IEEE Conference Paper Converter</h1>
                    <p className="hero__subtitle">
                        Transform your research into professionally formatted IEEE
                        conference papers with industry-standard precision.
                    </p>
                </header>
            )}

            {/* Main Content */}
            <main>
                {/* Upload Section */}
                {!isConverting && !isCompleted && (
                    <>
                        <div className="upload-wrapper animate-fadeIn" style={{ animationDelay: '0.1s' }}>
                            <FileUpload
                                onFileSelect={handleFileSelect}
                                selectedFile={file}
                                onRemove={() => setFile(null)}
                            />

                            {file && !isUploading && (
                                <div className="actions" style={{ marginTop: '2rem' }}>
                                    <button
                                        className="btn btn--primary"
                                        onClick={handleUpload}
                                        disabled={!file}
                                    >
                                        Start Conversion
                                    </button>
                                </div>
                            )}

                            {isUploading && (
                                <div className="actions" style={{ marginTop: '2rem' }}>
                                    <button className="btn btn--primary" disabled>
                                        <div className="spinner"></div>
                                        <span>Uploading...</span>
                                    </button>
                                </div>
                            )}

                            {error && (
                                <div className="error animate-fadeIn">
                                    <span>{error}</span>
                                </div>
                            )}
                        </div>

                        {/* Features Section */}
                        <section className="features animate-fadeIn" style={{ animationDelay: '0.2s' }}>
                            <div className="feature">
                                <h3 className="feature__title">Structure Detection</h3>
                                <p className="feature__text">
                                    Intelligent identification of academic components using
                                    specialized AI analysis.
                                </p>
                            </div>
                            <div className="feature">
                                <h3 className="feature__title">Precision Layout</h3>
                                <p className="feature__text">
                                    Full compliance with IEEE two-column standards and
                                    precise margin requirements.
                                </p>
                            </div>
                            <div className="feature">
                                <h3 className="feature__title">Vector Preservation</h3>
                                <p className="feature__text">
                                    High-fidelity extraction of complex diagrams and
                                    mathematical notation.
                                </p>
                            </div>
                            <div className="feature">
                                <h3 className="feature__title">Auto Bibliography</h3>
                                <p className="feature__text">
                                    Instant conversion to standard IEEE citation formats
                                    and alphabetical keyword sorting.
                                </p>
                            </div>
                        </section>
                    </>
                )}

                {/* Converting Section */}
                {isConverting && (
                    <div className="card animate-fadeIn">
                        <ConversionProgress
                            progress={status?.progress || 0}
                            message={status?.message || 'Processing...'}
                            currentStep={getStepFromProgress(status?.progress || 0)}
                        />
                    </div>
                )}

                {/* Completed Section */}
                {isCompleted && (
                    <div className="card animate-fadeIn">
                        <div className="success">
                            <span>Conversion completed successfully</span>
                        </div>

                        <DocumentPreview jobId={jobId} />

                        <div className="actions">
                            <DownloadButton jobId={jobId} />
                            <a
                                href={`/api/download-source/${jobId}`}
                                className="btn btn--secondary"
                                style={{ marginLeft: '1rem', textDecoration: 'none', display: 'inline-flex', alignItems: 'center' }}
                            >
                                📦 Source
                            </a>
                            <button className="btn btn--secondary" onClick={handleReset}>
                                Convert Another
                            </button>
                        </div>
                    </div>
                )}
            </main>
        </div>
    )
}

// Helper function to determine current step from progress
function getStepFromProgress(progress) {
    if (progress < 10) return 0
    if (progress < 30) return 1
    if (progress < 50) return 2
    if (progress < 70) return 3
    if (progress < 100) return 4
    return 5
}

export default App
