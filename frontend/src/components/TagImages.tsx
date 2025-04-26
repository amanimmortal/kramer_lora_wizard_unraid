import { useState, useEffect } from 'react';
import { Card, Image, Input, Button, message, Modal, InputNumber, Radio } from 'antd';
import { TagOutlined } from '@ant-design/icons';

interface TagImagesProps {
    projectId: string;
    triggerWord: string;
}

interface AutoTagSettings {
    label_type: 'tag' | 'caption';
    existing_tags_mode: 'ignore' | 'append' | 'overwrite';
    max_tags: number;
    min_threshold: number;
    blacklist_tags: string[];
    prepend_tags: string[];
    append_tags: string[];
}

export const TagImages = ({ projectId, triggerWord }: TagImagesProps) => {
    const [images, setImages] = useState<string[]>([]);
    const [tags, setTags] = useState<{ [key: string]: string }>({});
    const [isAutoTagging, setIsAutoTagging] = useState(false);
    const [autoTagModalVisible, setAutoTagModalVisible] = useState(false);
    const [autoTagSettings, setAutoTagSettings] = useState<AutoTagSettings>({
        label_type: "tag",
        existing_tags_mode: "ignore",
        max_tags: 10,
        min_threshold: 0.4,
        blacklist_tags: [],
        prepend_tags: [],
        append_tags: []
    });

    useEffect(() => {
        fetchImages();
    }, [projectId]);

    const fetchImages = async () => {
        try {
            const response = await fetch(`/api/training/images/${projectId}`);
            if (!response.ok) throw new Error('Failed to fetch images');
            const data = await response.json();
            setImages(data.images);
        } catch (error) {
            console.error('Error fetching images:', error);
            message.error('Failed to load images');
        }
    };

    const handleTagChange = (imageName: string, value: string) => {
        setTags(prev => ({ ...prev, [imageName]: value }));
    };

    const handleSaveTags = async (imageName: string) => {
        try {
            const response = await fetch(`/api/training/tags/${projectId}/${imageName}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tags: tags[imageName]?.split(',').map(t => t.trim()) || [],
                    trigger_word: triggerWord
                })
            });
            if (!response.ok) throw new Error('Failed to save tags');
            message.success('Tags saved successfully');
        } catch (error) {
            console.error('Error saving tags:', error);
            message.error('Failed to save tags');
        }
    };

    const handleAutoTag = async () => {
        try {
            setIsAutoTagging(true);
            message.loading('Auto-tagging images...', 0);

            const response = await fetch(`/api/training/${projectId}/auto-tag`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(autoTagSettings)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to auto-tag images');
            }

            const data = await response.json();

            // Update tags
            const newTags = { ...tags };
            Object.entries(data.results).forEach(([imageName, imageTags]) => {
                if (Array.isArray(imageTags)) {
                    newTags[imageName] = imageTags.join(', ');
                }
            });
            setTags(newTags);

            message.destroy();
            message.success('Auto-tagging completed successfully');
            setAutoTagModalVisible(false);
        } catch (error) {
            message.destroy();
            console.error('Auto-tag error:', error);
            message.error(error instanceof Error ? error.message : 'Failed to auto-tag images');
        } finally {
            setIsAutoTagging(false);
        }
    };

    return (
        <div className="space-y-4">
            <div className="mb-4 flex gap-2">
                <Button
                    type="primary"
                    icon={<TagOutlined />}
                    onClick={() => setAutoTagModalVisible(true)}
                    disabled={isAutoTagging}
                >
                    Auto Tag All
                </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {images.map(imageName => (
                    <Card key={imageName} className="w-full">
                        <Image
                            src={`/data/datasets/${projectId}/images/${imageName}`}
                            alt={imageName}
                            className="w-full h-48 object-cover mb-4"
                        />
                        <div className="flex gap-2 mb-2">
                            <Input.TextArea
                                value={tags[imageName] || ''}
                                onChange={e => handleTagChange(imageName, e.target.value)}
                                placeholder="Enter tags separated by commas"
                                autoSize={{ minRows: 2 }}
                                disabled={isAutoTagging}
                            />
                        </div>
                        <div className="flex gap-2">
                            <Button
                                type="primary"
                                onClick={() => handleSaveTags(imageName)}
                                disabled={isAutoTagging}
                            >
                                Save Tags
                            </Button>
                        </div>
                    </Card>
                ))}
            </div>

            <Modal
                title="Automatically label your images"
                open={autoTagModalVisible}
                onCancel={() => setAutoTagModalVisible(false)}
                onOk={handleAutoTag}
                okText="Auto Tag All"
                confirmLoading={isAutoTagging}
                width={600}
            >
                <div className="space-y-4">
                    <div>
                        <div className="mb-2">Label Type</div>
                        <Radio.Group
                            value={autoTagSettings.label_type}
                            onChange={e => setAutoTagSettings({ ...autoTagSettings, label_type: e.target.value })}
                            disabled={isAutoTagging}
                        >
                            <Radio.Button value="tag">Tag</Radio.Button>
                            <Radio.Button value="caption">Caption</Radio.Button>
                        </Radio.Group>
                    </div>

                    <div>
                        <div className="mb-2">Existing Tags</div>
                        <Radio.Group
                            value={autoTagSettings.existing_tags_mode}
                            onChange={e => setAutoTagSettings({ ...autoTagSettings, existing_tags_mode: e.target.value })}
                            disabled={isAutoTagging}
                        >
                            <Radio.Button value="ignore">Ignore</Radio.Button>
                            <Radio.Button value="append">Append</Radio.Button>
                            <Radio.Button value="overwrite">Overwrite</Radio.Button>
                        </Radio.Group>
                    </div>

                    <div>
                        <div className="mb-2">Max Tags</div>
                        <InputNumber
                            value={autoTagSettings.max_tags}
                            onChange={value => setAutoTagSettings({ ...autoTagSettings, max_tags: value || 10 })}
                            min={1}
                            max={50}
                            disabled={isAutoTagging}
                        />
                    </div>

                    <div>
                        <div className="mb-2">Min Threshold</div>
                        <InputNumber
                            value={autoTagSettings.min_threshold}
                            onChange={value => setAutoTagSettings({ ...autoTagSettings, min_threshold: value || 0.4 })}
                            min={0}
                            max={1}
                            step={0.1}
                            disabled={isAutoTagging}
                        />
                    </div>

                    <div>
                        <div className="mb-2">Blacklist (comma-separated)</div>
                        <Input
                            value={autoTagSettings.blacklist_tags.join(', ')}
                            onChange={e => setAutoTagSettings({
                                ...autoTagSettings,
                                blacklist_tags: e.target.value.split(',').map(t => t.trim()).filter(Boolean)
                            })}
                            placeholder="bad_tag_1, bad_tag_2"
                            disabled={isAutoTagging}
                        />
                    </div>

                    <div>
                        <div className="mb-2">Prepend Tags (comma-separated)</div>
                        <Input
                            value={autoTagSettings.prepend_tags.join(', ')}
                            onChange={e => setAutoTagSettings({
                                ...autoTagSettings,
                                prepend_tags: e.target.value.split(',').map(t => t.trim()).filter(Boolean)
                            })}
                            placeholder="important, details"
                            disabled={isAutoTagging}
                        />
                    </div>

                    <div>
                        <div className="mb-2">Append Tags (comma-separated)</div>
                        <Input
                            value={autoTagSettings.append_tags.join(', ')}
                            onChange={e => setAutoTagSettings({
                                ...autoTagSettings,
                                append_tags: e.target.value.split(',').map(t => t.trim()).filter(Boolean)
                            })}
                            placeholder="minor, details"
                            disabled={isAutoTagging}
                        />
                    </div>
                </div>
            </Modal>
        </div>
    );
}; 