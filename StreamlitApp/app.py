import streamlit as st
from model import ImageEnhancer
from PIL import Image
import time
from typing import Optional
import os 


# Custom CSS for professional appearance
st.markdown("""
<style>
    .header {
        background: linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .model-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .comparison-container {
        display: flex;
        justify-content: space-around;
        margin-top: 2rem;
        gap: 1rem;
    }
    .image-panel {
        flex: 1;
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
        font-size: 0.9rem;
        padding-top: 1rem;
        border-top: 1px solid #eaeaea;
    }
    .metric-card {
        background: #f1f3f5;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .stButton>button {
        background: linear-gradient(to right, #1a2a6c, #b21f1f);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state

if 'enhancer' not in st.session_state:
    st.session_state.enhancer = ImageEnhancer()
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        teacher_model_path = os.path.join(base_dir, "..", "models", "final_teacher_model.pth")
        student_model_path = os.path.join(base_dir, "..", "models", "final_student_model.pth")

        st.session_state.enhancer.load_models(
            teacher_path=teacher_model_path,
            student_path=student_model_path
        )
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# App header
st.markdown("""
<div class="header">
    <div class="title">REFINET</div>
    <div class="subtitle">Knowledge Distillation for Image Super-Resolution</div>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("## Configuration")
    model_type = st.radio(
        "Select Model",
        ('Student (Fast)', 'Teacher (High Quality)'),
        index=0
    )
    scale_factor = st.slider(
        "Upscaling Factor",
        min_value=2, max_value=8, value=4, step=1
    )
    
    st.markdown("---")
    st.markdown("### Model Information")
    st.markdown("""
    - **Teacher Model**: 
      - Parameters: 12.5M
      - Depth: 8 residual blocks
      - Recurrent refinement steps: 3
    - **Student Model**: 
      - Parameters: 1.2M (90% reduction)
      - Depth: 3 lightweight blocks
      - Recurrent refinement steps: 2 
    """)
    
    st.markdown("---")
    st.markdown("### About REFINET")
    st.markdown("""
    REFINET is a knowledge distillation framework for efficient image super-resolution:
    - Teacher model: High-precision MSAFN
    - Student model: Compact LightMSAFN
    - Trained on Vimeo90K dataset
    - Intel-optimized architecture
    """)

# Main content
st.markdown("## Image Enhancement")
input_image = None
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Input Image")
    uploaded_file = st.file_uploader(
        "Upload a low-resolution image",
        type=['jpg', 'jpeg', 'png'],
        key="uploader"
    )
    
    if uploaded_file is not None:
        input_image: Optional[Image.Image] = Image.open(uploaded_file)
        st.image(input_image, caption='Input Image', use_column_width=True)
    else:
        st.info("Please upload an image to begin enhancement")
        st.image("https://via.placeholder.com/512x512?text=Upload+Image", caption='Input Placeholder')
        # input_image: Optional[Image.Image] = None
with col2:
    st.markdown("### Enhanced Image")
    if st.button('Enhance Image', disabled=input_image is None, key="enhance_btn"):
        with st.spinner('Enhancing image...'):
            start_time = time.time()
            try:
                output_image = st.session_state.enhancer.enhance_image(
                    input_image,
                    'teacher' if model_type == 'Teacher (High Quality)' else 'student',
                    scale_factor
                )
                process_time = time.time() - start_time
                
                st.image(output_image, caption='Enhanced Image', use_column_width=True)
                
                # Performance metrics
                with st.expander("Performance Details"):
                    st.markdown("""
                    <div class="metric-card">
                        <h4>Enhancement Metrics</h4>
                    """, unsafe_allow_html=True)
                    
                    col_met1, col_met2 = st.columns(2)
                    with col_met1:
                        st.metric("Processing Time", f"{process_time:.2f} seconds")
                        st.metric("Model Type", model_type)
                    with col_met2:
                        st.metric("Upscale Factor", f"{scale_factor}x")
                        if input_image is not None:
                            st.metric("Input Resolution", f"{input_image.size[0]}*{input_image.size[1]}")       
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Download button
                from io import BytesIO
                buf = BytesIO()
                output_image.save(buf, format="PNG")
                st.download_button(
                    label="Download Enhanced Image",
                    data=buf.getvalue(),
                    file_name=f"REFINET_enhanced_{uploaded_file.name}",
                    mime="image/png",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Enhancement failed: {str(e)}")

# Model comparison section
st.markdown("---")
st.markdown("## Model Comparison")

col_compare1, col_compare2 = st.columns(2)
with col_compare1:
    st.markdown("### Teacher Model")
    st.markdown("""
    - High-precision reconstruction
    - Deeper architecture
    - Slower inference
    - Best for quality-critical applications
    """)
    if input_image and st.button('Run Teacher Model', key="teacher_btn"):
        with st.spinner('Running teacher model...'):
            start_time = time.time()
            teacher_output = st.session_state.enhancer.enhance_image(
                input_image, 'teacher', scale_factor
            )
            teacher_time = time.time() - start_time
            st.image(teacher_output, caption='Teacher Model Output', use_column_width=True)
            st.metric("Processing Time", f"{teacher_time:.2f} seconds")

with col_compare2:
    st.markdown("### Student Model")
    st.markdown("""
    - Real-time performance
    - Compact architecture
    - 90% parameter reduction
    - Best for resource-constrained devices
    """)
    if input_image and st.button('Run Student Model', key="student_btn"):
        with st.spinner('Running student model...'):
            start_time = time.time()
            student_output = st.session_state.enhancer.enhance_image(
                input_image, 'student', scale_factor
            )
            student_time = time.time() - start_time
            st.image(student_output, caption='Student Model Output', use_column_width=True)
            st.metric("Processing Time", f"{student_time:.2f} seconds")

# Technology showcase
st.markdown("---")
st.markdown("## Technology Highlights")
st.markdown("""
<div class="comparison-container">
    <div class="image-panel">
        <h4>Knowledge Distillation</h4>
        <p>Teacher model guides student training</p>
        <img src="https://miro.medium.com/v2/resize:fit:1400/1*T0WzA7IxYHxww_N5O0j66A.png" width="100%">
    </div>
    <div class="image-panel">
        <h4>Multi-Scale Processing</h4>
        <p>Captures features at multiple resolutions</p>
        <img src="https://www.researchgate.net/publication/350738642/figure/fig1/AS:1013560871591937@1618319283260/An-overview-of-the-proposed-multi-scale-feature-aggregation-network.png" width="100%">
    </div>
    <div class="image-panel">
        <h4>Recurrent Refinement</h4>
        <p>GRU-based iterative enhancement</p>
        <img src="https://miro.medium.com/v2/resize:fit:1000/1*T3J4k5Vl3HmXJhERWnV0Ug.png" width="100%">
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Intel Project Submission | Knowledge Distillation Framework</p>
    <p>© 2023 | REFINET Super-Resolution System</p>
</div>
""", unsafe_allow_html=True)