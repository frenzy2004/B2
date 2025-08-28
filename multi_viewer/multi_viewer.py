import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import nibabel as nib
import cv2
from scipy import ndimage
import tempfile
import os

def load_nifti_from_uploaded_file(uploaded_file):
    """Load NIfTI image from Streamlit UploadedFile and return data array"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load using nibabel
        nii = nib.load(tmp_path)
        data = nii.get_fdata()
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return data
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        return None

def load_sample_scan_files():
    """Load the sample brain scan files with different modalities"""
    try:
        scan_dir = "Brain MRI Sample Scan"
        if not os.path.exists(scan_dir):
            return None, None, None, None
        
        # Load different modalities
        t1_path = os.path.join(scan_dir, "BraTS20_Training_004_t1.nii")
        t2_path = os.path.join(scan_dir, "BraTS20_Training_004_t2.nii")
        flair_path = os.path.join(scan_dir, "BraTS20_Training_004_flair.nii")
        t1ce_path = os.path.join(scan_dir, "BraTS20_Training_004_t1ce.nii")
        
        # Load data
        t1_data = nib.load(t1_path).get_fdata() if os.path.exists(t1_path) else None
        t2_data = nib.load(t2_path).get_fdata() if os.path.exists(t2_path) else None
        flair_data = nib.load(flair_path).get_fdata() if os.path.exists(flair_path) else None
        t1ce_data = nib.load(t1ce_path).get_fdata() if os.path.exists(t1ce_path) else None
        
        return t1_data, t2_data, flair_data, t1ce_data
        
    except Exception as e:
        st.error(f"Error loading sample scan files: {str(e)}")
        return None, None, None, None

def normalize_image(image):
    """Normalize image to 0-255 range"""
    if image.max() == image.min():
        return np.zeros_like(image, dtype=np.uint8)
    normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    return normalized

def create_overlay_mask(segmentation, tumor_type_toggles):
    """Create overlay mask based on tumor type toggles"""
    overlay = np.zeros_like(segmentation, dtype=np.uint8)
    
    # BraTS segmentation label mapping
    tumor_regions = {
        'Necrotic': 1,     # NCR/NET - Non-enhancing tumor core
        'Edema': 2,        # ED - Peritumoral edematous tissue  
        'Enhancing': 4     # ET - GD-enhancing tumor
    }
    
    # Color mapping for display (1=red, 2=green, 3=blue in colormap)
    colors = {
        'Necrotic': 1,     # Red
        'Edema': 2,        # Green  
        'Enhancing': 3     # Blue
    }
    
    for tumor_type, enabled in tumor_type_toggles.items():
        if enabled and tumor_type in tumor_regions:
            # Create mask for this tumor type
            mask = (segmentation == tumor_regions[tumor_type])
            if np.any(mask):  # Only apply if mask has any true values
                overlay[mask] = colors[tumor_type]
    
    return overlay

def create_multi_view_display(t1_data, t2_data, flair_data, t1ce_data, segmentation_data, slice_idx, tumor_toggles):
    """Create 2x2 grid display of all modalities with overlays"""
    
    # Check if dark mode is enabled (from session state)
    dark_mode = st.session_state.get('dark_mode', True)
    
    # Set matplotlib style for dark mode
    if dark_mode:
        plt.style.use('dark_background')
        fig_facecolor = '#0e1117'
        text_color = 'white'
    else:
        plt.style.use('default')
        fig_facecolor = 'white'
        text_color = 'black'
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.patch.set_facecolor(fig_facecolor)
    fig.suptitle(f'Multi-Modality View - Slice {slice_idx}', fontsize=16, fontweight='bold', color=text_color)
    
    modalities = {
        'T1': (t1_data, axes[0, 0]),
        'T2': (t2_data, axes[0, 1]), 
        'FLAIR': (flair_data, axes[1, 0]),
        'T1CE': (t1ce_data, axes[1, 1])
    }
    
    # Color map for overlays
    colors = ['black', 'red', 'green', 'blue']  # 0=background, 1=necrotic, 2=edema, 3=enhancing
    overlay_cmap = ListedColormap(colors)
    
    for modality_name, (data, ax) in modalities.items():
        if data is not None:
            # Get the slice
            img_slice = data[:, :, slice_idx]
            normalized_slice = normalize_image(img_slice)
            
            # Display the base image
            ax.imshow(normalized_slice, cmap='gray', alpha=1.0)
            
            # Add overlay if segmentation exists
            if segmentation_data is not None:
                seg_slice = segmentation_data[:, :, slice_idx]
                overlay_mask = create_overlay_mask(seg_slice, tumor_toggles)
                
                # Only show overlay where mask > 0
                overlay_alpha = np.where(overlay_mask > 0, 0.4, 0.0)
                ax.imshow(overlay_mask, cmap=overlay_cmap, alpha=0.4, vmin=0, vmax=3)
            
            ax.set_title(modality_name, fontsize=14, fontweight='bold', color=text_color)
            ax.axis('off')
        else:
            # Adjust styling based on dark mode
            if dark_mode:
                text_bg = "darkgray"
                text_fg = "white"
            else:
                text_bg = "lightgray"
                text_fg = "black"
                
            ax.text(0.5, 0.5, f'{modality_name}\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, color=text_fg,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=text_bg))
            ax.axis('off')
    
    plt.tight_layout()
    return fig

def multi_viewer_tab():
    """Main function for multi-viewer tab"""
    st.header("🔬 Multi-Modality Viewer")
    st.write("View all MRI modalities simultaneously in a PACS-like layout with overlay controls.")
    
    # Try to load sample scan files with different modalities first
    t1_data, t2_data, flair_data, t1ce_data = load_sample_scan_files()
    
    if any([t1_data is not None, t2_data is not None, flair_data is not None, t1ce_data is not None]):
        st.success("✅ Using sample brain scan data with different modalities")
        segmentation_data = st.session_state.get('segmentation', None)
        data_loaded = True
        
    # Check if data exists in session state from main analysis as fallback
    elif 'mri_data' in st.session_state and st.session_state['mri_data'] is not None:
        st.warning("⚠️ Using single MRI data from main analysis tab (all modalities will look the same)")
        
        # Use data from session state as fallback
        mri_data = st.session_state['mri_data']
        segmentation_data = st.session_state.get('segmentation', None)
        
        # Fallback: use the same data for all modalities
        t1_data = mri_data
        t2_data = mri_data
        flair_data = mri_data
        t1ce_data = mri_data
        
        data_loaded = True
        
    else:
        st.warning("⚠️ No MRI data found. Please upload and process files in the main 'Upload & Process' tab first.")
        
        # Fallback: Allow manual upload for multi-viewer
        st.subheader("📁 Or Upload MRI Files Here")
        
        col1, col2 = st.columns(2)
        
        with col1:
            t1_file = st.file_uploader("T1 Image", type=['nii', 'nii.gz'], key="t1_multi")
            flair_file = st.file_uploader("FLAIR Image", type=['nii', 'nii.gz'], key="flair_multi")
        
        with col2:
            t2_file = st.file_uploader("T2 Image", type=['nii', 'nii.gz'], key="t2_multi")
            t1ce_file = st.file_uploader("T1CE Image", type=['nii', 'nii.gz'], key="t1ce_multi")
        
        seg_file = st.file_uploader("Segmentation (Optional)", type=['nii', 'nii.gz'], key="seg_multi")
        
        # Load data from uploaded files
        data_loaded = False
        t1_data = t2_data = flair_data = t1ce_data = segmentation_data = None
        
        if any([t1_file, t2_file, flair_file, t1ce_file]):
            with st.spinner("Loading MRI data..."):
                if t1_file:
                    t1_data = load_nifti_from_uploaded_file(t1_file)
                if t2_file:
                    t2_data = load_nifti_from_uploaded_file(t2_file)
                if flair_file:
                    flair_data = load_nifti_from_uploaded_file(flair_file)
                if t1ce_file:
                    t1ce_data = load_nifti_from_uploaded_file(t1ce_file)
                if seg_file:
                    segmentation_data = load_nifti_from_uploaded_file(seg_file)
                
                data_loaded = True
    
    if data_loaded:
        # Get dimensions from first available modality
        reference_data = next((data for data in [t1_data, t2_data, flair_data, t1ce_data] if data is not None), None)
        
        if reference_data is not None:
            max_slice = reference_data.shape[2] - 1
            
            # Controls section
            st.subheader("🎛️ Controls")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                slice_idx = st.slider("Select Slice", 0, max_slice, max_slice // 2, key="slice_multi")
            
            # Overlay toggles
            if segmentation_data is not None:
                st.subheader("🎨 Tumor Overlay Controls")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    necrotic_toggle = st.checkbox("🔴 Necrotic/Necrosis", value=True, key="necrotic_multi")
                
                with col2:
                    edema_toggle = st.checkbox("🟢 Edema", value=True, key="edema_multi")
                
                with col3:
                    enhancing_toggle = st.checkbox("🔵 Enhancing Tumor", value=True, key="enhancing_multi")
                
                tumor_toggles = {
                    'Necrotic': necrotic_toggle,
                    'Edema': edema_toggle,
                    'Enhancing': enhancing_toggle
                }
                
                # Legend
                st.markdown("""
                **Legend:**
                - 🔴 **Necrotic/Necrosis (NCR/NET)**: Non-enhancing tumor core
                - 🟢 **Edema (ED)**: Peritumoral edematous tissue
                - 🔵 **Enhancing Tumor (ET)**: Active tumor region
                """)
            else:
                tumor_toggles = {'Necrotic': False, 'Edema': False, 'Enhancing': False}
            
            # Display multi-view
            st.subheader("📊 Multi-Modality Display")
            
            with st.spinner("Generating multi-view display..."):
                fig = create_multi_view_display(
                    t1_data, t2_data, flair_data, t1ce_data, 
                    segmentation_data, slice_idx, tumor_toggles
                )
                
                st.pyplot(fig)
            
            # Additional info
            st.subheader("ℹ️ Image Information")
            
            info_cols = st.columns(4)
            modalities = [
                ("T1", t1_data), ("T2", t2_data), 
                ("FLAIR", flair_data), ("T1CE", t1ce_data)
            ]
            
            for i, (name, data) in enumerate(modalities):
                with info_cols[i]:
                    if data is not None:
                        st.metric(
                            f"{name} Shape", 
                            f"{data.shape[0]}×{data.shape[1]}×{data.shape[2]}"
                        )
                    else:
                        st.metric(f"{name} Shape", "N/A")
    
    else:
        st.info("👆 Please upload at least one MRI modality to start viewing.")
        
        # Show example layout
        st.subheader("📋 Expected Layout")
        st.markdown("""
        The multi-viewer will display your MRI scans in a **2×2 grid layout**:
        
        ```
        ┌─────────────┬─────────────┐
        │     T1      │     T2      │
        │             │             │
        ├─────────────┼─────────────┤
        │   FLAIR     │    T1CE     │
        │             │             │
        └─────────────┴─────────────┘
        ```
        
        **Features:**
        - ✅ Simultaneous view of all modalities
        - ✅ Synchronized slice navigation
        - ✅ Individual tumor overlay toggles
        - ✅ PACS-like viewing experience
        """)

if __name__ == "__main__":
    multi_viewer_tab()