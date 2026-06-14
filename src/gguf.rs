use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

const MAX_METADATA_ENTRIES: u64 = 1_000_000;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS: u64 = 10_000_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadata {
    pub path: PathBuf,
    pub file_size_bytes: u64,
    pub gguf_version: u32,
    pub name: Option<String>,
    pub architecture: String,
    pub block_count: u32,
    pub context_length: Option<u32>,
    pub embedding_length: Option<u32>,
    pub attention_head_count: Option<u32>,
    pub attention_head_count_kv: Option<u32>,
    pub attention_key_length: Option<u32>,
    pub attention_value_length: Option<u32>,
}

impl ModelMetadata {
    pub fn kv_dimensions_per_layer(&self) -> Option<(u64, u64)> {
        let embedding_length = u64::from(self.embedding_length?);
        let Some(head_count) = self.attention_head_count.map(u64::from) else {
            return Some((embedding_length, embedding_length));
        };
        if head_count == 0 {
            return None;
        }

        let kv_head_count = u64::from(self.attention_head_count_kv.unwrap_or(head_count as u32));
        if kv_head_count == 0 {
            return None;
        }
        let default_head_dimension = embedding_length / head_count;
        let key_length = u64::from(
            self.attention_key_length
                .unwrap_or(default_head_dimension as u32),
        );
        let value_length = u64::from(
            self.attention_value_length
                .unwrap_or(default_head_dimension as u32),
        );

        Some((kv_head_count * key_length, kv_head_count * value_length))
    }
}

#[derive(Debug)]
enum MetadataValue {
    Unsigned(u64),
    Signed(i64),
    String(String),
}

impl MetadataValue {
    fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Unsigned(value) => (*value).try_into().ok(),
            Self::Signed(value) => (*value).try_into().ok(),
            Self::String(_) => None,
        }
    }

    fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }
}

pub fn inspect(path: &Path) -> Result<ModelMetadata> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open GGUF model {}", path.display()))?;
    let file_size_bytes = file
        .metadata()
        .with_context(|| format!("failed to stat GGUF model {}", path.display()))?
        .len();

    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("{} is not a GGUF file", path.display());
    }

    let version = read_u32(&mut file)?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version {version}; expected version 2 or 3");
    }

    let _tensor_count = read_u64(&mut file)?;
    let metadata_count = read_u64(&mut file)?;
    if metadata_count > MAX_METADATA_ENTRIES {
        bail!("GGUF metadata count {metadata_count} exceeds the safety limit");
    }

    let mut metadata = HashMap::new();
    for _ in 0..metadata_count {
        let key = read_string(&mut file)?;
        let value_type = read_u32(&mut file)?;
        if let Some(value) = read_metadata_value(&mut file, value_type)? {
            metadata.insert(key, value);
        }
    }

    let architecture = string_value(&metadata, "general.architecture")
        .context("GGUF metadata is missing general.architecture")?;
    let prefix = format!("{architecture}.");
    let block_count = unsigned_value(&metadata, &(prefix.clone() + "block_count"))
        .context("GGUF metadata is missing the architecture block count")?;

    Ok(ModelMetadata {
        path: path.to_path_buf(),
        file_size_bytes,
        gguf_version: version,
        name: string_value(&metadata, "general.name"),
        architecture: architecture.clone(),
        block_count,
        context_length: unsigned_value(&metadata, &(prefix.clone() + "context_length")),
        embedding_length: unsigned_value(&metadata, &(prefix.clone() + "embedding_length")),
        attention_head_count: unsigned_value(&metadata, &(prefix.clone() + "attention.head_count")),
        attention_head_count_kv: unsigned_value(
            &metadata,
            &(prefix.clone() + "attention.head_count_kv"),
        ),
        attention_key_length: unsigned_value(&metadata, &(prefix.clone() + "attention.key_length")),
        attention_value_length: unsigned_value(&metadata, &(prefix + "attention.value_length")),
    })
}

fn string_value(metadata: &HashMap<String, MetadataValue>, key: &str) -> Option<String> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_string)
        .map(str::to_owned)
}

fn unsigned_value(metadata: &HashMap<String, MetadataValue>, key: &str) -> Option<u32> {
    metadata.get(key).and_then(MetadataValue::as_u32)
}

fn read_metadata_value(file: &mut File, value_type: u32) -> Result<Option<MetadataValue>> {
    Ok(match value_type {
        0 => Some(MetadataValue::Unsigned(u64::from(read_u8(file)?))),
        1 => Some(MetadataValue::Signed(i64::from(read_i8(file)?))),
        2 => Some(MetadataValue::Unsigned(u64::from(read_u16(file)?))),
        3 => Some(MetadataValue::Signed(i64::from(read_i16(file)?))),
        4 => Some(MetadataValue::Unsigned(u64::from(read_u32(file)?))),
        5 => Some(MetadataValue::Signed(i64::from(read_i32(file)?))),
        6 => {
            skip_bytes(file, 4)?;
            None
        }
        7 => Some(MetadataValue::Unsigned(u64::from(read_u8(file)?))),
        8 => Some(MetadataValue::String(read_string(file)?)),
        9 => {
            let element_type = read_u32(file)?;
            let element_count = read_u64(file)?;
            skip_array(file, element_type, element_count)?;
            None
        }
        10 => Some(MetadataValue::Unsigned(read_u64(file)?)),
        11 => Some(MetadataValue::Signed(read_i64(file)?)),
        12 => {
            skip_bytes(file, 8)?;
            None
        }
        _ => bail!("unsupported GGUF metadata value type {value_type}"),
    })
}

fn skip_array(file: &mut File, element_type: u32, element_count: u64) -> Result<()> {
    if element_count > MAX_ARRAY_ELEMENTS {
        bail!("GGUF metadata array exceeds the safety limit");
    }

    if let Some(width) = fixed_width(element_type) {
        return skip_bytes(
            file,
            element_count
                .checked_mul(width)
                .context("GGUF array byte count overflowed")?,
        );
    }

    for _ in 0..element_count {
        match element_type {
            8 => {
                let length = read_u64(file)?;
                if length > MAX_STRING_BYTES {
                    bail!("GGUF string exceeds the safety limit");
                }
                skip_bytes(file, length)?;
            }
            9 => {
                let nested_type = read_u32(file)?;
                let nested_count = read_u64(file)?;
                skip_array(file, nested_type, nested_count)?;
            }
            _ => bail!("unsupported GGUF array element type {element_type}"),
        }
    }
    Ok(())
}

fn fixed_width(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn read_string(file: &mut File) -> Result<String> {
    let length = read_u64(file)?;
    if length > MAX_STRING_BYTES {
        bail!("GGUF string exceeds the safety limit");
    }

    let mut bytes = vec![0_u8; length as usize];
    file.read_exact(&mut bytes)?;
    String::from_utf8(bytes).context("GGUF contains invalid UTF-8 metadata")
}

fn skip_bytes(file: &mut File, count: u64) -> Result<()> {
    let offset: i64 = count.try_into().context("GGUF skip offset is too large")?;
    file.seek(SeekFrom::Current(offset))?;
    Ok(())
}

fn read_u8(file: &mut File) -> Result<u8> {
    let mut bytes = [0_u8; 1];
    file.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_i8(file: &mut File) -> Result<i8> {
    Ok(read_u8(file)? as i8)
}

fn read_u16(file: &mut File) -> Result<u16> {
    let mut bytes = [0_u8; 2];
    file.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_i16(file: &mut File) -> Result<i16> {
    let mut bytes = [0_u8; 2];
    file.read_exact(&mut bytes)?;
    Ok(i16::from_le_bytes(bytes))
}

fn read_u32(file: &mut File) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(file: &mut File) -> Result<i32> {
    let mut bytes = [0_u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64(file: &mut File) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64(file: &mut File) -> Result<i64> {
    let mut bytes = [0_u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use tempfile::tempdir;

    use super::inspect;

    #[test]
    fn reads_planning_metadata_from_a_gguf_header() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("test.gguf");
        let mut file = File::create(&path).unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3_u32.to_le_bytes()).unwrap();
        file.write_all(&0_u64.to_le_bytes()).unwrap();
        file.write_all(&7_u64.to_le_bytes()).unwrap();
        write_string_value(&mut file, "general.architecture", "llama");
        write_string_value(&mut file, "general.name", "Test Model");
        write_u32_value(&mut file, "llama.block_count", 32);
        write_u32_value(&mut file, "llama.context_length", 8192);
        write_u32_value(&mut file, "llama.embedding_length", 4096);
        write_u32_value(&mut file, "llama.attention.head_count", 32);
        write_u32_value(&mut file, "llama.attention.head_count_kv", 8);
        drop(file);

        let model = inspect(&path).unwrap();

        assert_eq!(model.architecture, "llama");
        assert_eq!(model.block_count, 32);
        assert_eq!(model.context_length, Some(8192));
        assert_eq!(model.kv_dimensions_per_layer(), Some((1024, 1024)));
    }

    fn write_string_value(file: &mut File, key: &str, value: &str) {
        write_string(file, key);
        file.write_all(&8_u32.to_le_bytes()).unwrap();
        write_string(file, value);
    }

    fn write_u32_value(file: &mut File, key: &str, value: u32) {
        write_string(file, key);
        file.write_all(&4_u32.to_le_bytes()).unwrap();
        file.write_all(&value.to_le_bytes()).unwrap();
    }

    fn write_string(file: &mut File, value: &str) {
        file.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
        file.write_all(value.as_bytes()).unwrap();
    }
}
