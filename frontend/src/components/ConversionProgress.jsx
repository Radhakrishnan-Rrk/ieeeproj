const STEPS = [
    { label: 'Upload', icon: '📤' },
    { label: 'Parse', icon: '🔍' },
    { label: 'Analyze', icon: '🧠' },
    { label: 'Generate', icon: '📝' },
    { label: 'Compile', icon: '⚙️' },
    { label: 'Complete', icon: '✅' }
]

function ConversionProgress({ progress, message, currentStep }) {
    return (
        <div className="progress">
            {/* Progress bar */}
            <div className="progress__bar-container">
                <div
                    className="progress__bar"
                    style={{ width: `${progress}%` }}
                />
            </div>

            {/* Status */}
            <div className="progress__status">
                <span className="progress__message">{message}</span>
                <span className="progress__percentage">{progress}%</span>
            </div>

            {/* Steps indicator */}
            <div className="steps">
                {STEPS.map((step, index) => (
                    <div
                        key={step.label}
                        className={`step ${index < currentStep ? 'step--completed' :
                            index === currentStep ? 'step--active' : ''
                            }`}
                    >
                        <div className="step__icon">{step.icon}</div>
                        <span className="step__label">{step.label}</span>
                    </div>
                ))}
            </div>
        </div>
    )
}

export default ConversionProgress
