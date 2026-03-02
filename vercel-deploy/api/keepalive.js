export default async function handler(request, response) {
  try {
    // Ping the Streamlit app to keep it awake
    const res = await fetch('https://resonair-jwwizveqwck36qcxphzkkk.streamlit.app/');
    if (res.ok) {
      return response.status(200).json({ status: 'Streamlit app is awake!' });
    } else {
      return response.status(res.status).json({ status: 'Pinged, but received non-200 status' });
    }
  } catch (error) {
    return response.status(500).json({ status: 'Failed to ping', error: error.message });
  }
}
