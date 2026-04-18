import React from "react";
import {UseGetKeyModels} from "@/hooks/keys/useGetKeyModels";
import { Card, Tag } from 'antd';

interface KeyModelListProps {
    key_id: string;
}

const extractDefaultTags = (source: string) => {
    if (source === 'all-proxy-models') {
        return <Tag className="ml-2">All proxy models</Tag>
    } else if (source === 'all-team-models') {
        return <Tag className="ml-2">All team models</Tag>
    }
    return ''
}



const KeyModelList: React.FC<KeyModelListProps> = ({ key_id }) => {
    const { data: keyModels, isLoading} = UseGetKeyModels(key_id)

    

    const title = keyModels ? <>Model {extractDefaultTags(keyModels.source)}</> : 'Model'
    return (
        <Card title={title} loading={isLoading}>
            <div className="mt-2 flex flex-wrap gap-2">
                {keyModels && keyModels.models.map((item:string)=> {return <Tag>{item}</Tag>})}
            </div>
        </Card>
    );
};

export default KeyModelList;

