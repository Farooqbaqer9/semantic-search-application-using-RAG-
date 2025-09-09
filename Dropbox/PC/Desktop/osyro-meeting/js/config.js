// API Configuration for Oysro Meeting Room
const API_CONFIG = {
    // Backend API base URL
    // To switch environments, comment or uncomment the appropriate BASE_URL line below:
    // For local development, use:
    //BASE_URL: 'http://localhost:5000', // <-- Uncomment this line for local development
    // For production, use:
     BASE_URL: 'https://your-backend-domain.com', // <-- Uncomment this line for production and comment out the local line above
    // Only one BASE_URL line should be active (not commented) at a time.
    
    // API endpoints
    ENDPOINTS: {
        ENROLL: '/api/enroll',
        PROCESS_MEETING: '/api/process_meeting',
        SPEAKERS: '/api/speakers',
        MEETINGS: '/api/meetings',
        MEETING_DETAILS: '/api/meeting',
        END_MEETING: '/api/end_meeting',
        TEST_AUDIO: '/api/test_audio'
    },
    
    // Audio settings
    AUDIO: {
        SAMPLE_RATE: 16000,
        CHANNELS: 1,
        FORMAT: 'wav'
    },
    
    // Meeting settings
    MEETING: {
        AUTO_SAVE_INTERVAL: 30000, // 30 seconds
        MAX_RECORDING_TIME: 3600000 // 1 hour
    }
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = API_CONFIG;
}