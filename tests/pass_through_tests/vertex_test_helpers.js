function isVertexQuotaError(error) {
    const message = [
        error && error.message,
        error && error.stack,
        error && error.cause && JSON.stringify(error.cause),
    ].filter(Boolean).join('\n');

    return (
        message.includes('429') ||
        message.includes('Too Many Requests') ||
        message.includes('RESOURCE_EXHAUSTED')
    );
}

async function runVertexRequestOrSkip(requestFn) {
    try {
        return await requestFn();
    } catch (error) {
        if (isVertexQuotaError(error)) {
            console.warn('Vertex AI quota exhausted; skipping live provider assertions for this run');
            return null;
        }
        throw error;
    }
}

module.exports = { isVertexQuotaError, runVertexRequestOrSkip };
