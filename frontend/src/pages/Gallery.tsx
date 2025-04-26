import { useState } from 'react';
import { Input, Select } from 'antd';
import { TagImages } from '../components/TagImages';
import { useParams } from 'react-router-dom';

const { Option } = Select;

export const Gallery = () => {
    const { projectId } = useParams();
    const [triggerWord, setTriggerWord] = useState('');
    const [tagMode, setTagMode] = useState('danbooru');

    return (
        <div className="p-4">
            <div className="mb-4 flex gap-4 items-center">
                <Input
                    placeholder="Trigger word"
                    value={triggerWord}
                    onChange={e => setTriggerWord(e.target.value)}
                    className="max-w-xs"
                />
                <Select
                    value={tagMode}
                    onChange={setTagMode}
                    className="w-40"
                >
                    <Option value="danbooru">Danbooru Tags</Option>
                    <Option value="caption">BLIP Caption</Option>
                </Select>
            </div>
            <TagImages
                projectId={projectId || ''}
                triggerWord={triggerWord}
            />
        </div>
    );
}; 